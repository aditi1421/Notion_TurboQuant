"""
Interactive demo: Search your Notion docs with TurboQuant.

Shows side-by-side results: FP16 brute-force vs TurboQuant compressed search.
"""

import torch
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.index import TurboQuantIndex


def load_data(data_dir: str = "data"):
    """Load embeddings and metadata."""
    embeddings = np.load(os.path.join(data_dir, "notion_embeddings.npy"))
    metadata = []
    with open(os.path.join(data_dir, "notion_metadata.jsonl")) as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))
    return torch.tensor(embeddings).float(), metadata


def run_demo(data_dir: str = "data", model_name: str = "all-MiniLM-L6-v2"):
    """Interactive search demo."""
    from sentence_transformers import SentenceTransformer

    print("=" * 70)
    print("TurboQuant Notion Search Demo")
    print("=" * 70)

    # Load data
    print("\nLoading embeddings...")
    vectors, metadata = load_data(data_dir)
    dim = vectors.shape[1]
    n = vectors.shape[0]
    print(f"  {n} document chunks, {dim} dimensions")

    # Build indices
    print("\nBuilding indices...")
    tq3 = TurboQuantIndex(dim, bits=3)
    tq3.add(vectors, metadata, keep_raw=True)
    mem = tq3.memory_usage()
    print(f"  TurboQuant 3-bit: {mem['compressed_bytes']/1024:.1f} KB ({mem['compression_ratio']:.1f}x compression)")
    print(f"  FP16 baseline:    {mem['fp16_bytes']/1024:.1f} KB")

    # Load embedding model for queries
    print(f"\nLoading query encoder: {model_name}")
    encoder = SentenceTransformer(model_name)

    # Interactive loop
    print("\n" + "=" * 70)
    print("Type a search query (or 'quit' to exit)")
    print("=" * 70)

    while True:
        try:
            query_text = input("\nQuery: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query_text or query_text.lower() in ("quit", "exit", "q"):
            break

        # Embed query
        query_vec = encoder.encode([query_text], normalize_embeddings=True)
        query = torch.tensor(query_vec).float().squeeze(0)

        # Search both methods
        tq_results = tq3.search(query, top_k=5)
        bf_results = tq3.brute_force_search(query, top_k=5)

        # Display side by side
        print(f"\n{'FP16 Brute Force (Ground Truth)':<40} | {'TurboQuant 3-bit (Compressed)':<40}")
        print("-" * 83)

        for i in range(5):
            bf = bf_results[i] if i < len(bf_results) else {}
            tq = tq_results[i] if i < len(tq_results) else {}

            bf_title = bf.get("title", "—")[:35]
            bf_score = bf.get("score", 0)
            tq_title = tq.get("title", "—")[:35]
            tq_score = tq.get("score", 0)

            match = "✓" if bf.get("index") == tq.get("index") else "✗"

            print(f"  {i+1}. {bf_title:<33} {bf_score:>5.3f} | {match} {tq_title:<33} {tq_score:>5.3f}")

        # Recall for this query
        bf_ids = {r["index"] for r in bf_results}
        tq_ids = {r["index"] for r in tq_results}
        recall = len(bf_ids & tq_ids) / len(bf_ids) if bf_ids else 0
        print(f"\n  Recall@5: {recall:.0%} | Compression: {mem['compression_ratio']:.1f}x")

    print("\nDone.")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    run_demo(data_dir=data_dir)
