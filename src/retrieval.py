from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence

import faiss
import numpy as np
from rank_bm25 import BM25Okapi


class EncoderProtocol(Protocol):
    def encode(self, texts: Sequence[str], **kwargs: Any) -> Any:
        ...


class HybridRetriever:
    """Hybrid lexical + dense retriever over in-memory chunks."""

    def __init__(
        self,
        chunks: Sequence[Dict[str, Any]],
        *,
        semantic_weight: float = 0.5,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encoder: Optional[EncoderProtocol] = None,
        use_dense: bool = True,
        raise_on_embedding_error: bool = False,
    ) -> None:
        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be between 0 and 1.")

        self.semantic_weight = semantic_weight
        self.embedding_model_name = embedding_model_name
        self.encoder = encoder
        self.use_dense = use_dense
        self.raise_on_embedding_error = raise_on_embedding_error

        self._chunks = self._dedupe_chunks(chunks)
        self._texts = [str(chunk.get("text", "")) for chunk in self._chunks]
        self._tokenized_corpus = [self._tokenize(text) for text in self._texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._chunks else None

        self._dense_index: Optional[faiss.IndexFlatIP] = None
        self._dense_embeddings: Optional[np.ndarray] = None
        self._dense_enabled = bool(self.use_dense)

    @staticmethod
    def _dedupe_chunks(chunks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen_ids = set()
        for index, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id")
            stable_id = chunk_id if chunk_id is not None else f"chunk-{index}"
            if stable_id in seen_ids:
                continue
            seen_ids.add(stable_id)
            normalized = dict(chunk)
            normalized["chunk_id"] = stable_id
            deduped.append(normalized)
        return deduped

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        if np.isclose(max_score, min_score):
            if np.isclose(max_score, 0.0):
                return np.zeros_like(scores)
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def _load_default_encoder(self) -> EncoderProtocol:
        if not self.embedding_model_name:
            raise ValueError("embedding_model_name must be provided when encoder is not injected.")
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.embedding_model_name)

    def _get_encoder(self) -> EncoderProtocol:
        if self.encoder is None:
            self.encoder = self._load_default_encoder()
        return self.encoder

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        encoder = self._get_encoder()
        try:
            vectors = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        except TypeError:
            vectors = encoder.encode(texts)
        vectors_np = np.asarray(vectors, dtype=np.float32)
        if vectors_np.ndim != 2:
            raise ValueError("Encoder output must be a 2D array-like structure.")
        return vectors_np

    def _ensure_dense_index(self) -> bool:
        if not self._dense_enabled or not self._chunks:
            return False
        if self._dense_index is not None and self._dense_embeddings is not None:
            return True
        try:
            embeddings = self._encode(self._texts)
            embeddings = self._normalize_embeddings(embeddings).astype(np.float32)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            self._dense_embeddings = embeddings
            self._dense_index = index
            return True
        except Exception:
            self._dense_enabled = False
            if self.raise_on_embedding_error:
                raise
            return False

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._chunks or top_k <= 0:
            return []
        if not isinstance(query, str) or not query.strip():
            return []

        top_k = min(top_k, len(self._chunks))
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        bm25_scores = (
            np.asarray(self._bm25.get_scores(query_tokens), dtype=np.float32)
            if self._bm25 is not None
            else np.zeros(len(self._chunks), dtype=np.float32)
        )
        bm25_norm = self._normalize_scores(bm25_scores)

        dense_scores = np.zeros(len(self._chunks), dtype=np.float32)
        dense_norm = dense_scores.copy()
        dense_available = self._ensure_dense_index()
        if dense_available and self._dense_index is not None:
            query_embedding = self._encode([query])
            query_embedding = self._normalize_embeddings(query_embedding).astype(np.float32)
            similarities, _ = self._dense_index.search(query_embedding, len(self._chunks))
            dense_scores = similarities[0]
            dense_norm = self._normalize_scores(dense_scores)

        bm25_weight = 1.0 - self.semantic_weight
        dense_weight = self.semantic_weight if dense_available else 0.0
        fused_scores = (bm25_weight * bm25_norm) + (dense_weight * dense_norm)

        ranked_indices = sorted(
            range(len(self._chunks)),
            key=lambda idx: (fused_scores[idx], bm25_norm[idx], dense_norm[idx], -idx),
            reverse=True,
        )[:top_k]

        results: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            result = dict(self._chunks[idx])
            result["score"] = float(fused_scores[idx])
            result["bm25_score"] = float(bm25_scores[idx])
            result["dense_score"] = float(dense_scores[idx])
            results.append(result)
        return results
