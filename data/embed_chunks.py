"""
Embed text chunks using sentence-transformers.

Generates embedding vectors from Notion document chunks,
saved as numpy arrays for benchmarking.
"""

import numpy as np
import json
import os
from typing import List, Dict, Any


def load_chunks(path: str) -> List[Dict[str, Any]]:
    """Load chunks from JSONL file."""
    chunks = []
    with open(path) as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def embed_chunks(
    chunks: List[Dict[str, Any]],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    output_dir: str = "data",
) -> np.ndarray:
    """
    Embed chunks using sentence-transformers.

    Args:
        chunks: list of chunk dicts with 'text' field
        model_name: sentence-transformers model name
            - "all-MiniLM-L6-v2" → 384d, fast, good for demo
            - "BAAI/bge-base-en-v1.5" → 768d, better quality
        batch_size: encoding batch size
        output_dir: directory to save outputs

    Returns:
        embeddings: (N, dim) numpy array
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity
    )

    # Save embeddings
    os.makedirs(output_dir, exist_ok=True)
    emb_path = os.path.join(output_dir, "notion_embeddings.npy")
    np.save(emb_path, embeddings)
    print(f"Saved embeddings: {emb_path} — shape {embeddings.shape}")

    # Save metadata
    meta_path = os.path.join(output_dir, "notion_metadata.jsonl")
    with open(meta_path, 'w') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
    print(f"Saved metadata: {meta_path}")

    # Generate test queries from page titles
    titles = list({c["title"] for c in chunks if c.get("title")})
    queries_path = os.path.join(output_dir, "test_queries.json")
    with open(queries_path, 'w') as f:
        json.dump(titles, f, indent=2)
    print(f"Saved {len(titles)} test queries: {queries_path}")

    return embeddings


def embed_queries(
    queries: List[str],
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Embed query strings."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(queries, normalize_embeddings=True)


if __name__ == "__main__":
    import sys
    chunks_path = sys.argv[1] if len(sys.argv) > 1 else "data/notion_chunks.jsonl"
    chunks = load_chunks(chunks_path)
    embed_chunks(chunks)
