"""
TurboQuant adapted for embedding vector search.

Wraps TurboQuantProd for document embeddings instead of KV cache vectors.
Key difference: embeddings are typically unit-normalized and higher-dimensional
(384, 768, 1536) vs KV head_dim (128).

The asymmetric inner product estimator is the killer feature for search:
queries stay FP16, only the corpus is compressed.
"""

import torch
import math
from typing import Optional, Dict, List

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant.turboquant import TurboQuantProd, TurboQuantMSE


class EmbeddingQuantizer:
    """
    Compresses embedding vectors using TurboQuant and supports
    asymmetric similarity search (FP16 query × compressed corpus).
    """

    def __init__(self, dim: int, bits: int = 3, seed: int = 42, device: str = "cpu"):
        """
        Args:
            dim: embedding dimension (e.g., 384, 768, 1536)
            bits: total bits per coordinate (3 recommended — 5x compression, >95% recall)
            seed: random seed for reproducibility
            device: torch device
        """
        self.dim = dim
        self.bits = bits
        self.device = device

        # TurboQuantProd: (bits-1) for MSE + 1 for QJL = total bits per coord
        self.quantizer = TurboQuantProd(dim, bits, seed=seed, device=device)

    @torch.no_grad()
    def compress(self, vectors: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compress a batch of embedding vectors.

        Args:
            vectors: (N, dim) float tensor, ideally L2-normalized

        Returns:
            Compressed representation dict with:
                - mse_indices: (N, dim) codebook indices
                - qjl_signs: (N, dim) sign bits
                - residual_norm: (N,) residual L2 norms
                - n_vectors: int
        """
        vectors = vectors.to(self.device).float()

        # Normalize if not already (most embedding models output unit vectors)
        norms = torch.norm(vectors, dim=-1, keepdim=True)
        normalized = vectors / (norms + 1e-8)

        compressed = self.quantizer.quantize(normalized)
        compressed["vector_norms"] = norms.squeeze(-1)
        compressed["n_vectors"] = vectors.shape[0]
        return compressed

    @torch.no_grad()
    def compress_batch(self, vectors: torch.Tensor, batch_size: int = 1024) -> Dict[str, torch.Tensor]:
        """
        Compress large corpus in batches to avoid OOM.

        Args:
            vectors: (N, dim) float tensor
            batch_size: vectors per batch

        Returns:
            Single compressed dict with all vectors concatenated
        """
        all_compressed = []
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            all_compressed.append(self.compress(batch))

        # Concatenate all batches
        return {
            "mse_indices": torch.cat([c["mse_indices"] for c in all_compressed], dim=0),
            "qjl_signs": torch.cat([c["qjl_signs"] for c in all_compressed], dim=0),
            "residual_norm": torch.cat([c["residual_norm"] for c in all_compressed], dim=0),
            "vector_norms": torch.cat([c["vector_norms"] for c in all_compressed], dim=0),
            "n_vectors": sum(c["n_vectors"] for c in all_compressed),
        }

    @torch.no_grad()
    def asymmetric_search(
        self,
        query: torch.Tensor,
        compressed_corpus: Dict[str, torch.Tensor],
        top_k: int = 10,
    ) -> tuple:
        """
        Asymmetric similarity search: FP16 query × compressed corpus.

        The query is NOT compressed — only the corpus is. This is what makes
        TurboQuant viable for search with minimal recall loss.

        Args:
            query: (dim,) or (Q, dim) float tensor (uncompressed query embeddings)
            compressed_corpus: dict from compress() or compress_batch()
            top_k: number of results to return

        Returns:
            (scores, indices) — top_k scores and their corpus indices
        """
        query = query.to(self.device).float()
        if query.dim() == 1:
            query = query.unsqueeze(0)

        # Normalize query
        q_norm = torch.norm(query, dim=-1, keepdim=True)
        query_normalized = query / (q_norm + 1e-8)

        # Reconstruct MSE component
        x_mse = self.quantizer.mse.dequantize(compressed_corpus["mse_indices"])  # (N, dim)

        # Term 1: <query, x_mse> (standard matmul)
        term1 = query_normalized @ x_mse.T  # (Q, N)

        # Term 2: QJL correction for unbiased inner product
        q_projected = query_normalized @ self.quantizer.S.T  # (Q, qjl_dim)
        qjl_ip = q_projected @ compressed_corpus["qjl_signs"].T  # (Q, N)

        m = self.quantizer.qjl_dim
        correction_scale = math.sqrt(math.pi / 2) / m
        term2 = correction_scale * qjl_ip * compressed_corpus["residual_norm"].unsqueeze(0)

        # Combine: unbiased inner product estimate on unit vectors, scaled by corpus norms
        scores = (term1 + term2) * compressed_corpus["vector_norms"].unsqueeze(0)

        # Also scale by query norms for true inner product
        scores = scores * q_norm

        # Top-k
        k = min(top_k, scores.shape[1])
        top_scores, top_indices = torch.topk(scores, k, dim=-1)

        return top_scores.squeeze(0), top_indices.squeeze(0)

    def memory_usage_bytes(self, n_vectors: int) -> Dict[str, float]:
        """Estimate memory usage for n compressed vectors."""
        mse_bits = self.quantizer.mse_bits
        dim = self.dim

        mse_bytes = n_vectors * dim * mse_bits / 8
        qjl_bytes = n_vectors * dim * 1 / 8  # 1 bit per sign
        norm_bytes = n_vectors * 2 * 2  # residual_norm + vector_norm, FP16 each
        total = mse_bytes + qjl_bytes + norm_bytes

        fp16_bytes = n_vectors * dim * 2  # baseline

        return {
            "compressed_bytes": total,
            "fp16_bytes": fp16_bytes,
            "compression_ratio": fp16_bytes / total,
            "bits_per_dim": (total * 8) / (n_vectors * dim),
        }
