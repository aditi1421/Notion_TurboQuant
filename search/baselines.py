"""
Baseline search methods for comparison with TurboQuant.
"""

import torch
import time
import numpy as np
from typing import Dict, List, Tuple, Any


class FP16BruteForce:
    """Standard FP16 brute-force search (what Turbopuffer effectively does per-namespace)."""

    def __init__(self, dim: int, device: str = "cpu"):
        self.dim = dim
        self.device = device
        self.vectors = None
        self.metadata = []

    def add(self, vectors: torch.Tensor, metadata: List[Dict[str, Any]]):
        self.vectors = vectors.to(self.device).float()
        self.metadata = metadata

    def search(self, query: torch.Tensor, top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        query = query.to(self.device).float()
        if query.dim() == 1:
            query = query.unsqueeze(0)
        scores = query @ self.vectors.T
        k = min(top_k, scores.shape[1])
        return torch.topk(scores.squeeze(0), k)

    def memory_bytes(self) -> float:
        if self.vectors is None:
            return 0
        return self.vectors.numel() * 2  # FP16

    def __len__(self):
        return self.vectors.shape[0] if self.vectors is not None else 0


class BinaryQuantization:
    """
    Binary quantization baseline — each dimension stored as 1 bit.
    Turbopuffer supports this as a fast pre-filter.
    """

    def __init__(self, dim: int, device: str = "cpu"):
        self.dim = dim
        self.device = device
        self.binary_vectors = None
        self.metadata = []

    def add(self, vectors: torch.Tensor, metadata: List[Dict[str, Any]]):
        self.binary_vectors = (vectors > 0).to(torch.int8).to(self.device)
        self.metadata = metadata

    def search(self, query: torch.Tensor, top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        query = query.to(self.device).float()
        if query.dim() == 1:
            query = query.unsqueeze(0)
        query_binary = (query > 0).to(torch.int8)
        # Hamming similarity = number of matching bits
        scores = (query_binary @ self.binary_vectors.T).float().squeeze(0)
        k = min(top_k, scores.shape[0])
        return torch.topk(scores, k)

    def memory_bytes(self) -> float:
        if self.binary_vectors is None:
            return 0
        return self.binary_vectors.numel() / 8  # 1 bit per dim

    def __len__(self):
        return self.binary_vectors.shape[0] if self.binary_vectors is not None else 0


class ProductQuantization:
    """
    Product Quantization baseline using faiss.

    PQ is what most vector databases (including Turbopuffer) use internally.
    Requires codebook training on the corpus — TurboQuant doesn't.
    """

    def __init__(self, dim: int, n_subquantizers: int = 8, n_bits: int = 8, device: str = "cpu"):
        self.dim = dim
        self.n_subquantizers = n_subquantizers
        self.n_bits = n_bits
        self.device = device
        self.index = None
        self.metadata = []
        self._raw_vectors = None

    def add(self, vectors: torch.Tensor, metadata: List[Dict[str, Any]]):
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu required for PQ baseline: pip install faiss-cpu")

        vectors_np = vectors.numpy().astype(np.float32)
        self._raw_vectors = vectors_np

        self.index = faiss.IndexPQ(self.dim, self.n_subquantizers, self.n_bits)
        self.index.train(vectors_np)
        self.index.add(vectors_np)
        self.metadata = metadata

    def search(self, query: torch.Tensor, top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        query_np = query.numpy().astype(np.float32)
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)

        distances, indices = self.index.search(query_np, top_k)
        # faiss returns L2 distances; convert to similarity (negative distance)
        scores = torch.tensor(-distances[0])
        indices = torch.tensor(indices[0])
        return scores, indices

    def memory_bytes(self) -> float:
        if self.index is None:
            return 0
        n = self.index.ntotal
        # PQ stores n_subquantizers bytes per vector + codebook
        return n * self.n_subquantizers + self.n_subquantizers * (2**self.n_bits) * (self.dim // self.n_subquantizers) * 4

    def __len__(self):
        return self.index.ntotal if self.index is not None else 0


def benchmark_method(search_fn, queries: torch.Tensor, top_k: int = 10) -> Dict[str, float]:
    """Time a search method over multiple queries."""
    latencies = []
    for q in queries:
        start = time.perf_counter()
        search_fn(q, top_k)
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    return {
        "avg_latency_ms": np.mean(latencies),
        "p50_latency_ms": np.median(latencies),
        "p99_latency_ms": np.percentile(latencies, 99),
        "n_queries": len(queries),
    }
