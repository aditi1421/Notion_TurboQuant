"""
Vector index using TurboQuant compression.

Drop-in replacement concept for a vector search index (like Turbopuffer),
but with TurboQuant-compressed storage and asymmetric search.
"""

import torch
import json
from typing import Dict, List, Optional, Any

from .embedding_quantizer import EmbeddingQuantizer


class TurboQuantIndex:
    """
    A simple vector index that stores compressed embeddings and supports
    asymmetric similarity search.

    Mirrors Turbopuffer's namespace-per-workspace model.
    """

    def __init__(self, dim: int, bits: int = 3, device: str = "cpu"):
        self.dim = dim
        self.bits = bits
        self.device = device
        self.quantizer = EmbeddingQuantizer(dim, bits, device=device)

        self.compressed = None
        self.metadata: List[Dict[str, Any]] = []
        self._raw_vectors: Optional[torch.Tensor] = None  # kept for recall comparison

    def add(self, vectors: torch.Tensor, metadata: List[Dict[str, Any]], keep_raw: bool = True):
        """
        Add vectors with metadata to the index.

        Args:
            vectors: (N, dim) float tensor
            metadata: list of N dicts (e.g., {"title": ..., "page_id": ..., "chunk": ...})
            keep_raw: if True, store raw vectors for recall comparison (disable in production)
        """
        assert vectors.shape[0] == len(metadata), "vectors and metadata must have same length"
        assert vectors.shape[1] == self.dim, f"expected dim={self.dim}, got {vectors.shape[1]}"

        self.compressed = self.quantizer.compress_batch(vectors)
        self.metadata = metadata

        if keep_raw:
            self._raw_vectors = vectors.to(self.device).float()

    def search(self, query: torch.Tensor, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search the index with an uncompressed query.

        Returns list of top_k results with scores and metadata.
        """
        if self.compressed is None:
            return []

        scores, indices = self.quantizer.asymmetric_search(query, self.compressed, top_k)

        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            result = {"score": score, "index": idx}
            if idx < len(self.metadata):
                result.update(self.metadata[idx])
            results.append(result)
        return results

    def brute_force_search(self, query: torch.Tensor, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        FP16 brute-force search (baseline for recall comparison).
        Only works if keep_raw=True was passed to add().
        """
        if self._raw_vectors is None:
            raise ValueError("Raw vectors not stored — pass keep_raw=True to add()")

        query = query.to(self.device).float()
        if query.dim() == 1:
            query = query.unsqueeze(0)

        scores = query @ self._raw_vectors.T  # (1, N)
        k = min(top_k, scores.shape[1])
        top_scores, top_indices = torch.topk(scores, k, dim=-1)

        results = []
        for score, idx in zip(top_scores.squeeze(0).tolist(), top_indices.squeeze(0).tolist()):
            result = {"score": score, "index": idx}
            if idx < len(self.metadata):
                result.update(self.metadata[idx])
            results.append(result)
        return results

    def memory_usage(self) -> Dict[str, float]:
        """Report memory usage stats."""
        n = self.compressed["n_vectors"] if self.compressed else 0
        return self.quantizer.memory_usage_bytes(n)

    def recall_at_k(self, queries: torch.Tensor, k: int = 10) -> Dict[str, float]:
        """
        Measure recall@k: what fraction of FP16 top-k results appear
        in TurboQuant top-k results.
        """
        if self._raw_vectors is None:
            raise ValueError("Need raw vectors for recall measurement")

        if queries.dim() == 1:
            queries = queries.unsqueeze(0)

        recalls = {f"recall@{k}": 0.0}
        for kk in [1, 5, k]:
            matches = 0
            total = 0
            for q in queries:
                tq_results = self.search(q, top_k=kk)
                bf_results = self.brute_force_search(q, top_k=kk)

                tq_ids = {r["index"] for r in tq_results}
                bf_ids = {r["index"] for r in bf_results}

                matches += len(tq_ids & bf_ids)
                total += len(bf_ids)

            recalls[f"recall@{kk}"] = matches / total if total > 0 else 0.0

        return recalls

    def __len__(self):
        return self.compressed["n_vectors"] if self.compressed else 0

    def __repr__(self):
        n = len(self)
        mem = self.memory_usage()
        return (
            f"TurboQuantIndex(dim={self.dim}, bits={self.bits}, "
            f"n_vectors={n}, compression={mem['compression_ratio']:.1f}x)"
        )
