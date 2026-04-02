"""
Face Recognizer — FAISS-backed ArcFace Embedding Store

Stores 512-dimensional ArcFace embeddings in a FAISS IndexFlatIP index.
Cosine similarity is computed as inner product on L2-normalised vectors.

Storage layout:
  - FAISS IndexFlatIP (normalised vectors → cosine similarity via inner product)
  - int64 FAISS id → student_id string (stored in _faiss_id_to_student dict)
  - Persisted by pickling the raw {student_id: [embedding, ...]} dict; the
    FAISS index is rebuilt from that dict on load.

Threshold guide (cosine similarity, 0–1, higher = stricter match):
    0.35 — lenient  |  0.40 — default  |  0.55 — strict
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

# NOTE: Do NOT call logging.basicConfig() here.
# Logging configuration is the responsibility of the application entry point (main.py).
logger = logging.getLogger(__name__)

EMBED_DIM = 512   # ArcFace output dimension


class FaceRecognizer:
    """
    ArcFace 512-d embedding store backed by FAISS IndexFlatIP.

    Cosine similarity is computed as inner product on L2-normalised vectors.
    Threshold guide (cosine similarity, 0–1, higher = stricter):
        0.35 lenient | 0.40 default | 0.55 strict
    """

    def __init__(self,
                 model_name: str = "ArcFace",
                 distance_metric: str = "cosine",
                 recognition_threshold: float = 0.40):
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.recognition_threshold = recognition_threshold

        self._build_empty_index()

        # raw storage for save/load (FAISS index is not directly serialisable)
        # {student_id: [embedding_512d, ...]}
        self._raw: Dict[str, List[np.ndarray]] = {}

        logger.info(f"FaceRecognizer (ArcFace + FAISS) ready — threshold={recognition_threshold}")

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _build_empty_index(self):
        flat = faiss.IndexFlatIP(EMBED_DIM)
        self._index = faiss.IndexIDMap(flat)
        self._id_counter: int = 0
        self._faiss_id_to_student: Dict[int, str] = {}

    def _rebuild_index(self):
        """Rebuild FAISS index from self._raw (called after load)."""
        self._build_empty_index()
        for student_id, embeddings in self._raw.items():
            for emb in embeddings:
                self._add_to_index(student_id, emb)

    def _normalise(self, v: np.ndarray) -> np.ndarray:
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)

    def _add_to_index(self, student_id: str, embedding: np.ndarray):
        vec = self._normalise(embedding).reshape(1, EMBED_DIM)
        faiss_id = self._id_counter
        self._index.add_with_ids(vec, np.array([faiss_id], dtype=np.int64))
        self._faiss_id_to_student[faiss_id] = student_id
        self._id_counter += 1

    # ------------------------------------------------------------------
    # Embedding management
    # ------------------------------------------------------------------

    def add_embedding(self, student_id: str, embedding: np.ndarray) -> bool:
        """Store a pre-computed ArcFace 512-d embedding for a student."""
        try:
            self._raw.setdefault(student_id, []).append(embedding)
            self._add_to_index(student_id, embedding)
            logger.info(
                f"Added ArcFace embedding for '{student_id}' "
                f"(total stored: {len(self._raw[student_id])})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add embedding for '{student_id}': {e}")
            return False

    # DEPRECATED — raw images cannot be embedded without the InsightFace pipeline.
    # Use add_embedding() with a pre-computed ArcFace vector instead.
    def add_face_to_database(self, student_id: str, face_image: np.ndarray) -> bool:
        """Legacy shim — not supported in ArcFace+FAISS mode. Always returns False."""
        logger.warning(
            "add_face_to_database() called with raw image — "
            "use add_embedding() with a pre-computed ArcFace vector."
        )
        return False

    def remove_face_from_database(self, student_id: str) -> bool:
        if student_id not in self._raw:
            logger.warning(f"'{student_id}' not in database.")
            return False
        del self._raw[student_id]
        # Rebuild index without this student
        self._rebuild_index()
        logger.info(f"Removed '{student_id}' and rebuilt FAISS index.")
        return True

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def recognize_from_embedding(self, query_embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        1:N identification using FAISS inner product search (cosine similarity).
        Returns the best match if similarity >= threshold.
        """
        if self._index.ntotal == 0:
            logger.warning("FAISS index is empty.")
            return None

        query = self._normalise(query_embedding).reshape(1, EMBED_DIM)
        # Search for top-1 nearest vector
        D, I = self._index.search(query, k=1)
        best_score = float(D[0][0])
        best_faiss_id = int(I[0][0])

        if best_faiss_id == -1 or best_score < self.recognition_threshold:
            logger.info(f"No match — best_score={best_score:.4f} < {self.recognition_threshold}")
            return None

        student_id = self._faiss_id_to_student[best_faiss_id]
        logger.info(f"Recognised '{student_id}' (cosine={best_score:.4f})")
        return {
            "student_id": student_id,
            "confidence": best_score,
            "distance": 1.0 - best_score,
            "matched": True,
        }

    # DEPRECATED — raw images are not supported; use recognize_from_embedding() instead.
    def recognize_face(self, face_image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Legacy interface — not supported in ArcFace+FAISS mode. Always returns None."""
        logger.warning(
            "recognize_face() called with raw image — "
            "use recognize_from_embedding() with a pre-computed ArcFace vector."
        )
        return None

    # ------------------------------------------------------------------
    # Verification (1:1)
    # ------------------------------------------------------------------

    def verify_from_embedding(self, query_embedding: np.ndarray, student_id: str) -> Tuple[bool, float]:
        vecs = self._raw.get(student_id)
        if not vecs:
            return False, 0.0
        q = self._normalise(query_embedding)
        scores = [float(np.dot(q, self._normalise(v))) for v in vecs]
        score = float(np.mean(scores))
        return score >= self.recognition_threshold, score

    # DEPRECATED — raw images are not supported; use verify_from_embedding() instead.
    def verify_face(self, face_image: np.ndarray, student_id: str) -> Tuple[bool, float]:
        """Legacy interface — not supported in ArcFace+FAISS mode. Always returns (False, 0.0)."""
        return False, 0.0

    # ------------------------------------------------------------------
    # Persistence (pickle raw vectors; rebuild FAISS on load)
    # ------------------------------------------------------------------

    def save_embeddings(self, file_path: str) -> bool:
        try:
            path = Path(file_path).with_suffix(".pkl")
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self._raw, f)
            logger.info(f"Saved {len(self._raw)} identities → {path}")
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
                self._raw = pickle.load(f)
            self._rebuild_index()
            logger.info(f"Loaded {len(self._raw)} identities, rebuilt FAISS index ({self._index.ntotal} vectors)")
            return True
        except Exception as e:
            logger.error(f"Load error: {e}")
            return False

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_database_stats(self) -> Dict[str, Any]:
        return {
            "total_faces": len(self._raw),
            "total_vectors": self._index.ntotal,
            "student_ids": list(self._raw.keys()),
            "model_name": self.model_name,
            "distance_metric": self.distance_metric,
            "threshold": self.recognition_threshold,
            "embedding_dim": EMBED_DIM,
            "index_backend": "FAISS IndexFlatIP",
        }

    @property
    def is_trained(self) -> bool:
        return self._index.ntotal > 0
