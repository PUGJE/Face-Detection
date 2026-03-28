"""
Face Recognition Pipeline — InsightFace (RetinaFace + ArcFace)

Detection:   RetinaFace (ONNX) — handles any angle, low light, partial occlusion
Recognition: ArcFace MobileFaceNet (ONNX) — 512-d embeddings, ~99.4% LFW accuracy

No TensorFlow. Runs on ONNX Runtime (CPU or GPU).

Author: Face Recognition Team
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from backend.ml.face_detection import FaceDetector      # kept as Haar fallback
from backend.ml.face_recognition import FaceRecognizer
from backend.ml.anti_spoofing import AntiSpoofingDetector
from backend.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# InsightFace lazy singleton — shared across all pipeline instances
# ---------------------------------------------------------------------------
_insightface_app = None

def _get_insightface():
    global _insightface_app
    if _insightface_app is not None:
        return _insightface_app
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _insightface_app = app
        logger.info("InsightFace (RetinaFace + ArcFace) initialised successfully.")
    except Exception as e:
        logger.error(f"InsightFace failed to load: {e}. Falling back to Haar+HOG.")
        _insightface_app = None
    return _insightface_app


class FaceRecognitionPipeline:
    """
    End-to-end face recognition pipeline.

    Primary path  : InsightFace (RetinaFace detection + ArcFace recognition)
    Fallback path : OpenCV Haar cascade detection + HOG cosine similarity
    """

    def __init__(self,
                 detection_confidence: float = None,
                 recognition_threshold: float = None,
                 model_name: str = "ArcFace",
                 enable_anti_spoofing: bool = True):

        if detection_confidence is None:
            detection_confidence = settings.face_detection_confidence
        if recognition_threshold is None:
            recognition_threshold = settings.face_recognition_threshold

        # Haar cascade kept for fallback standalone detection
        self.detector = FaceDetector(min_detection_confidence=detection_confidence)

        # ArcFace-based recognizer (stores 512-d embeddings)
        self.recognizer = FaceRecognizer(
            model_name=model_name,
            distance_metric="cosine",
            recognition_threshold=recognition_threshold,
        )

        # Anti-spoofing
        self.enable_anti_spoofing = enable_anti_spoofing
        if enable_anti_spoofing:
            self.anti_spoof = AntiSpoofingDetector(
                texture_threshold=0.60,
                motion_threshold=0.4,
            )
            logger.info("Anti-spoofing enabled")
        else:
            self.anti_spoof = None

        # Pre-load InsightFace eagerly so the first request isn't slow
        _get_insightface()

        logger.info("FaceRecognitionPipeline initialised (InsightFace primary)")

    # ------------------------------------------------------------------
    # Helper: run InsightFace on a full image
    # ------------------------------------------------------------------
    def _get_insightface_faces(self, image: np.ndarray):
        """Return InsightFace face objects sorted by detection score (desc)."""
        app = _get_insightface()
        if app is None:
            return []
        try:
            faces = app.get(image)
            return sorted(faces, key=lambda f: f.det_score, reverse=True)
        except Exception as e:
            logger.error(f"InsightFace inference error: {e}")
            return []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_student(self, student_id: str, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect the most prominent face in *image* and store its ArcFace embedding.
        """
        faces = self._get_insightface_faces(image)

        if not faces:
            # Fallback: check if Haar finds a face at least
            haar_face = self.detector.detect_single_face(image)
            if haar_face is None:
                return {"success": False, "error": "No face detected in image", "student_id": student_id}
            return {"success": False, "error": "Face detected but ArcFace model unavailable", "student_id": student_id}

        best = faces[0]
        embedding = best.embedding  # 512-d float32 ArcFace vector

        success = self.recognizer.add_embedding(student_id, embedding)

        if success:
            bbox = best.bbox.astype(int).tolist()
            return {
                "success": True,
                "student_id": student_id,
                "face_confidence": float(best.det_score),
                "bbox": bbox,
                "embedding_dim": len(embedding),
            }
        return {"success": False, "error": "Failed to store embedding", "student_id": student_id}

    # ------------------------------------------------------------------
    # Recognition (1:N)
    # ------------------------------------------------------------------
    def recognize_student(self, image: np.ndarray) -> Dict[str, Any]:
        faces = self._get_insightface_faces(image)

        if not faces:
            return {"success": False, "error": "No face detected", "recognized": False}

        best = faces[0]
        result = self.recognizer.recognize_from_embedding(best.embedding)

        if result is None:
            return {
                "success": True,
                "recognized": False,
                "error": "Face not recognised",
                "detection_confidence": float(best.det_score),
            }

        return {
            "success": True,
            "recognized": True,
            "student_id": result["student_id"],
            "recognition_confidence": result["confidence"],
            "recognition_distance": result["distance"],
            "detection_confidence": float(best.det_score),
            "bbox": best.bbox.astype(int).tolist(),
        }

    # ------------------------------------------------------------------
    # Recognition with liveness check
    # ------------------------------------------------------------------
    def recognize_with_liveness(self, image: np.ndarray) -> Dict[str, Any]:
        faces = self._get_insightface_faces(image)

        if not faces:
            return {"success": False, "error": "No face detected", "recognized": False, "is_live": False}

        best = faces[0]
        bbox_tuple = tuple(best.bbox.astype(int).tolist())  # (x1,y1,x2,y2)
        # Convert to (x, y, w, h) for anti-spoof which uses that format
        x1, y1, x2, y2 = bbox_tuple
        haar_bbox = (x1, y1, x2 - x1, y2 - y1)

        # Liveness check
        if self.enable_anti_spoofing and self.anti_spoof:
            liveness_result = self.anti_spoof.check_liveness(image, face_region=haar_bbox)
            if not liveness_result.get("is_live", False):
                return {
                    "success": True,
                    "recognized": False,
                    "is_live": False,
                    "liveness_score": liveness_result.get("liveness_score", 0),
                    "error": "Liveness check failed — possible spoofing",
                    "detection_confidence": float(best.det_score),
                }
            liveness_score = liveness_result.get("liveness_score", 1.0)
        else:
            liveness_score = 1.0

        result = self.recognizer.recognize_from_embedding(best.embedding)

        if result is None:
            return {
                "success": True,
                "recognized": False,
                "is_live": True,
                "liveness_score": liveness_score,
                "error": "Face not recognised",
                "detection_confidence": float(best.det_score),
            }

        return {
            "success": True,
            "recognized": True,
            "is_live": True,
            "liveness_score": liveness_score,
            "student_id": result["student_id"],
            "recognition_confidence": result["confidence"],
            "recognition_distance": result["distance"],
            "detection_confidence": float(best.det_score),
            "bbox": best.bbox.astype(int).tolist(),
        }

    # ------------------------------------------------------------------
    # Verification (1:1)
    # ------------------------------------------------------------------
    def verify_student(self, student_id: str, image: np.ndarray) -> Dict[str, Any]:
        faces = self._get_insightface_faces(image)
        if not faces:
            return {"success": False, "verified": False, "error": "No face detected"}

        is_verified, score = self.recognizer.verify_from_embedding(faces[0].embedding, student_id)
        return {
            "success": True,
            "verified": is_verified,
            "student_id": student_id,
            "distance": 1.0 - score,
            "detection_confidence": float(faces[0].det_score),
        }

    # ------------------------------------------------------------------
    # Batch frame processing (attendance scanning)
    # ------------------------------------------------------------------
    def process_attendance_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        faces = self._get_insightface_faces(frame)
        results = []
        for face in faces:
            rec = self.recognizer.recognize_from_embedding(face.embedding)
            x1, y1, x2, y2 = face.bbox.astype(int).tolist()
            bbox = (x1, y1, x2 - x1, y2 - y1)
            if rec:
                results.append({
                    "recognized": True,
                    "student_id": rec["student_id"],
                    "confidence": rec["confidence"],
                    "distance": rec["distance"],
                    "bbox": bbox,
                })
            else:
                results.append({
                    "recognized": False,
                    "bbox": bbox,
                    "detection_confidence": float(face.det_score),
                })
        return results

    # ------------------------------------------------------------------
    # Database persistence
    # ------------------------------------------------------------------
    def save_database(self, file_path: str = None) -> bool:
        if file_path is None:
            file_path = f"{settings.embeddings_path}/face_embeddings.pkl"
        return self.recognizer.save_embeddings(file_path)

    def load_database(self, file_path: str = None) -> bool:
        if file_path is None:
            file_path = f"{settings.embeddings_path}/face_embeddings.pkl"
        return self.recognizer.load_embeddings(file_path)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "detector": {
                "backend": "InsightFace RetinaFace (ONNX)" if _get_insightface() else "OpenCV Haar Cascade",
                "confidence_threshold": self.detector.min_detection_confidence,
            },
            "recognizer": self.recognizer.get_database_stats(),
        }

    # ------------------------------------------------------------------
    # Visual overlay helpers (for debugging / demo)
    # ------------------------------------------------------------------
    def draw_recognition_results(self, image: np.ndarray,
                                 results: List[Dict[str, Any]]) -> np.ndarray:
        out = image.copy()
        for r in results:
            bbox = r["bbox"]
            x, y, w, h = bbox
            if r["recognized"]:
                color = (0, 255, 0)
                label = f"{r['student_id']} ({r['confidence']:.2f})"
            else:
                color = (0, 0, 255)
                label = "Unknown"
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            cv2.putText(out, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out
