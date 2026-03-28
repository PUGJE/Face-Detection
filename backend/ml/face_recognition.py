"""
Face Recognition Module — ArcFace Embeddings (via InsightFace ONNX)

Stores 512-d ArcFace embeddings and performs cosine-similarity matching.
No TensorFlow required — runs purely on ONNX Runtime.

Accuracy: ~99.4% on LFW benchmark (ArcFace / MobileFaceNet)

Author: Face Recognition Team
"""

import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class FaceRecognizer:
    """
    Face Recognizer using ArcFace 512-d embeddings + cosine similarity.

    Embeddings are generated externally by InsightFace (recognition_pipeline.py)
    and stored here. On recognition, the query embedding is compared against all
    stored embeddings using cosine similarity.

    Threshold guide (cosine similarity, higher = stricter):
        0.35 - lenient  (more matches, slight false-positive risk)
        0.50 - balanced (default, works well in practice)
        0.65 - strict   (fewer false positives, may miss occluded faces)
    """

    def __init__(self,
                 model_name: str = "ArcFace",
                 distance_metric: str = "cosine",
                 recognition_threshold: float = 0.40):
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.recognition_threshold = recognition_threshold

        # {student_id: [embedding_512d_normalized, ...]}
        self.database: Dict[str, List[np.ndarray]] = {}

        logger.info(
            f"FaceRecognizer (ArcFace) ready — threshold={recognition_threshold}"
        )

    # ------------------------------------------------------------------
    # Embedding management
    # ------------------------------------------------------------------

    def add_embedding(self, student_id: str, embedding: np.ndarray) -> bool:
        """Store a pre-computed ArcFace 512-d embedding for a student."""
        try:
            norm = np.linalg.norm(embedding)
            emb = embedding / (norm + 1e-9)
            self.database.setdefault(student_id, []).append(emb)
            logger.info(
                f"Stored ArcFace embedding for '{student_id}' "
                f"(total vectors: {len(self.database[student_id])})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store embedding for '{student_id}': {e}")
            return False

    def add_face_to_database(self, student_id: str, face_image: np.ndarray) -> bool:
        """
        Legacy shim — raw face images are handled upstream by the pipeline.
        This path should not normally be reached.
        """
        logger.warning(
            "add_face_to_database() called with raw image — ArcFace embedding "
            "must be extracted by the pipeline. Call add_embedding() instead."
        )
        return False

    def remove_face_from_database(self, student_id: str) -> bool:
        if student_id in self.database:
            del self.database[student_id]
            logger.info(f"Removed '{student_id}' from database.")
            return True
        logger.warning(f"'{student_id}' not found in database.")
        return False

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def recognize_from_embedding(self, query_embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        1:N identification from a pre-computed ArcFace embedding.

        Returns the best-matching student if their average similarity across
        all stored vectors exceeds the threshold.
        """
        if not self.database:
            logger.warning("Database is empty.")
            return None

        query = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        best_id, best_score = None, -1.0

        for student_id, embeddings in self.database.items():
            scores = [_cosine_similarity(query, e) for e in embeddings]
            score = float(np.mean(scores))
            if score > best_score:
                best_score = score
                best_id = student_id

        if best_score >= self.recognition_threshold:
            logger.info(f"Recognised '{best_id}' (similarity={best_score:.4f})")
            return {
                "student_id": best_id,
                "confidence": best_score,
                "distance": 1.0 - best_score,
                "matched": True,
            }

        logger.info(f"No match — best={best_score:.4f} < threshold={self.recognition_threshold}")
        return None

    def recognize_face(self, face_image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Legacy interface kept for compatibility with old callers.
        Builds a rough HOG embedding as fallback when ArcFace isn't available.
        """
        # If database has real ArcFace embeddings (512-d), this path won't work
        # gracefully. The pipeline always calls recognize_from_embedding() directly.
        logger.warning("recognize_face() called with raw image — prefer recognize_from_embedding()")
        return None

    def verify_from_embedding(self, query_embedding: np.ndarray, student_id: str) -> Tuple[bool, float]:
        """1:1 verification for a specific student."""
        embeddings = self.database.get(student_id)
        if not embeddings:
            return False, 0.0
        query = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        score = float(np.mean([_cosine_similarity(query, e) for e in embeddings]))
        return score >= self.recognition_threshold, score

    def verify_face(self, face_image: np.ndarray, student_id: str) -> Tuple[bool, float]:
        """Legacy shim."""
        return False, 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_embeddings(self, file_path: str) -> bool:
        try:
            path = Path(file_path).with_suffix(".pkl")
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self.database, f)
            logger.info(f"Saved {len(self.database)} identities → {path}")
            return True
        except Exception as e:
            logger.error(f"Save error: {e}")
            return False

    def load_embeddings(self, file_path: str) -> bool:
        try:
            path = Path(file_path).with_suffix(".pkl")
            if not path.exists():
                logger.warning(f"No embeddings file at {path}")
                return False
            with open(path, "rb") as f:
                self.database = pickle.load(f)
            logger.info(f"Loaded {len(self.database)} identities from {path}")
            return True
        except Exception as e:
            logger.error(f"Load error: {e}")
            return False

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_database_stats(self) -> Dict[str, Any]:
        return {
            "total_faces": len(self.database),
            "student_ids": list(self.database.keys()),
            "model_name": self.model_name,
            "distance_metric": self.distance_metric,
            "threshold": self.recognition_threshold,
            "embedding_dim": 512,
        }

    @property
    def is_trained(self) -> bool:
        return len(self.database) > 0
