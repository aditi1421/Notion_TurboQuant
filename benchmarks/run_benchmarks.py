"""
Benchmark TurboQuant vs baselines on Notion workspace embeddings.

Measures: recall@k, query latency, compression ratio, index build time.
"""

import torch
import numpy as np
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.index import TurboQuantIndex
from search.baselines import FP16BruteForce, BinaryQuantization, ProductQuantization, benchmark_method


def load_data(data_dir: str = "data"):
    """Load embeddings and metadata."""
    embeddings = np.load(os.path.join(data_dir, "notion_embeddings.npy"))
    metadata = []
    with open(os.path.join(data_dir, "notion_metadata.jsonl")) as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))

    with open(os.path.join(data_dir, "test_queries.json")) as f:
        query_texts = json.load(f)

    return embeddings, metadata, query_texts


def generate_query_embeddings(query_texts, model_name="all-MiniLM-L6-v2"):
    """Embed test queries."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(query_texts, normalize_embeddings=True)


def run_benchmarks(data_dir: str = "data", output_dir: str = "benchmarks"):
    """Run full benchmark suite."""
    print("=" * 70)
    print("TurboQuant for Notion Search — Benchmark Suite")
    print("=" * 70)

    # Load data
    print("\nLoading Notion embeddings...")
    embeddings, metadata, query_texts = load_data(data_dir)
    vectors = torch.tensor(embeddings).float()
    dim = vectors.shape[1]
    n_vectors = vectors.shape[0]
    print(f"  Corpus: {n_vectors} vectors, {dim} dimensions")

    # Generate query embeddings
    print(f"\nEmbedding {len(query_texts)} test queries...")
    query_embeddings = generate_query_embeddings(query_texts)
    queries = torch.tensor(query_embeddings).float()
    print(f"  Queries: {queries.shape[0]} × {queries.shape[1]}")

    results = {}

    # --- FP16 Brute Force (Ground Truth) ---
    print("\n--- FP16 Brute Force (Ground Truth) ---")
    fp16 = FP16BruteForce(dim)
    t0 = time.perf_counter()
    fp16.add(vectors, metadata)
    build_time_fp16 = time.perf_counter() - t0

    latency_fp16 = benchmark_method(lambda q, k: fp16.search(q, k), queries)
    results["FP16"] = {
        "build_time_ms": build_time_fp16 * 1000,
        "memory_bytes": fp16.memory_bytes(),
        "bits_per_dim": 16.0,
        "compression_ratio": 1.0,
        **latency_fp16,
    }
    print(f"  Memory: {fp16.memory_bytes() / 1024:.1f} KB")
    print(f"  Avg latency: {latency_fp16['avg_latency_ms']:.2f} ms")

    # --- Ground truth top-k for recall ---
    print("\nComputing ground truth top-k...")
    gt_top_k = {}
    for k in [1, 5, 10]:
        gt_top_k[k] = []
        for q in queries:
            scores, indices = fp16.search(q, top_k=k)
            gt_top_k[k].append(set(indices.tolist()))

    # --- TurboQuant at various bit-widths ---
    for bits in [2, 3, 4]:
        print(f"\n--- TurboQuant {bits}-bit ---")
        tq = TurboQuantIndex(dim, bits=bits)

        t0 = time.perf_counter()
        tq.add(vectors, metadata, keep_raw=False)
        build_time = time.perf_counter() - t0

        mem = tq.memory_usage()
        latency = benchmark_method(lambda q, k: tq.search(q, k), queries)

        # Recall
        recalls = {}
        for k in [1, 5, 10]:
            matches = 0
            total = 0
            for i, q in enumerate(queries):
                tq_results = tq.search(q, top_k=k)
                tq_ids = {r["index"] for r in tq_results}
                matches += len(tq_ids & gt_top_k[k][i])
                total += len(gt_top_k[k][i])
            recalls[f"recall@{k}"] = matches / total if total > 0 else 0.0

        results[f"TQ-{bits}bit"] = {
            "build_time_ms": build_time * 1000,
            "memory_bytes": mem["compressed_bytes"],
            "bits_per_dim": mem["bits_per_dim"],
            "compression_ratio": mem["compression_ratio"],
            **latency,
            **recalls,
        }

        print(f"  Compression: {mem['compression_ratio']:.1f}x ({mem['bits_per_dim']:.1f} bits/dim)")
        print(f"  Memory: {mem['compressed_bytes'] / 1024:.1f} KB")
        print(f"  Build time: {build_time * 1000:.1f} ms")
        print(f"  Avg latency: {latency['avg_latency_ms']:.2f} ms")
        for k_val in [1, 5, 10]:
            print(f"  Recall@{k_val}: {recalls[f'recall@{k_val}']:.1%}")

    # --- Binary Quantization ---
    print("\n--- Binary Quantization ---")
    bq = BinaryQuantization(dim)
    t0 = time.perf_counter()
    bq.add(vectors, metadata)
    build_time_bq = time.perf_counter() - t0

    latency_bq = benchmark_method(lambda q, k: bq.search(q, k), queries)

    recalls_bq = {}
    for k in [1, 5, 10]:
        matches = 0
        total = 0
        for i, q in enumerate(queries):
            scores, indices = bq.search(q, top_k=k)
            bq_ids = set(indices.tolist())
            matches += len(bq_ids & gt_top_k[k][i])
            total += len(gt_top_k[k][i])
        recalls_bq[f"recall@{k}"] = matches / total if total > 0 else 0.0

    results["Binary"] = {
        "build_time_ms": build_time_bq * 1000,
        "memory_bytes": bq.memory_bytes(),
        "bits_per_dim": 1.0,
        "compression_ratio": n_vectors * dim * 2 / bq.memory_bytes() if bq.memory_bytes() > 0 else 0,
        **latency_bq,
        **recalls_bq,
    }
    print(f"  Compression: {results['Binary']['compression_ratio']:.1f}x")
    print(f"  Avg latency: {latency_bq['avg_latency_ms']:.2f} ms")
    for k_val in [1, 5, 10]:
        print(f"  Recall@{k_val}: {recalls_bq[f'recall@{k_val}']:.1%}")

    # --- Product Quantization ---
    try:
        print("\n--- Product Quantization (faiss) ---")
        n_sub = min(8, dim // 4)  # faiss needs dim % n_sub == 0
        while dim % n_sub != 0 and n_sub > 1:
            n_sub -= 1

        # PQ needs at least 256 training points for 8-bit codebooks
        if n_vectors < 256:
            print(f"  Skipped — PQ requires ≥256 training vectors, have {n_vectors}")
            print(f"  (This is a KEY advantage of TurboQuant: zero training needed!)")
            raise ImportError("skip")
        pq = ProductQuantization(dim, n_subquantizers=n_sub, n_bits=8)
        t0 = time.perf_counter()
        pq.add(vectors, metadata)
        build_time_pq = time.perf_counter() - t0

        latency_pq = benchmark_method(lambda q, k: pq.search(q, k), queries)

        recalls_pq = {}
        for k in [1, 5, 10]:
            matches = 0
            total = 0
            for i, q in enumerate(queries):
                scores, indices = pq.search(q, top_k=k)
                pq_ids = set(indices.tolist())
                matches += len(pq_ids & gt_top_k[k][i])
                total += len(gt_top_k[k][i])
            recalls_pq[f"recall@{k}"] = matches / total if total > 0 else 0.0

        results["PQ"] = {
            "build_time_ms": build_time_pq * 1000,
            "memory_bytes": pq.memory_bytes(),
            "bits_per_dim": pq.memory_bytes() * 8 / (n_vectors * dim),
            "compression_ratio": n_vectors * dim * 2 / pq.memory_bytes() if pq.memory_bytes() > 0 else 0,
            **latency_pq,
            **recalls_pq,
        }
        print(f"  Build time: {build_time_pq * 1000:.1f} ms (includes codebook training)")
        print(f"  Avg latency: {latency_pq['avg_latency_ms']:.2f} ms")
        for k_val in [1, 5, 10]:
            print(f"  Recall@{k_val}: {recalls_pq[f'recall@{k_val}']:.1%}")
    except ImportError:
        print("  Skipped (faiss-cpu not installed)")

    # --- Summary Table ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = f"{'Method':<15} {'Bits/dim':>9} {'Compress':>9} {'Recall@1':>9} {'Recall@5':>9} {'Recall@10':>10} {'Latency':>10} {'Build':>10}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:<15} "
            f"{r.get('bits_per_dim', 0):>8.1f}b "
            f"{r.get('compression_ratio', 0):>8.1f}x "
            f"{r.get('recall@1', 1.0):>8.1%} "
            f"{r.get('recall@5', 1.0):>8.1%} "
            f"{r.get('recall@10', 1.0):>9.1%} "
            f"{r.get('avg_latency_ms', 0):>8.2f}ms "
            f"{r.get('build_time_ms', 0):>8.1f}ms"
        )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.json")
    # Convert numpy types for JSON serialization
    clean_results = {}
    for k, v in results.items():
        clean_results[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv for kk, vv in v.items()}
    with open(results_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


def notion_scale_projection(results: dict, data_dir: str = "data"):
    """Project costs at Notion's 10B vector scale."""
    embeddings = np.load(os.path.join(data_dir, "notion_embeddings.npy"))
    dim = embeddings.shape[1]

    print("\n" + "=" * 70)
    print("NOTION-SCALE COST PROJECTION (10B vectors)")
    print("=" * 70)

    # S3 pricing: ~$0.023/GB/month
    s3_price_per_gb = 0.023
    n_vectors = 10_000_000_000  # 10 billion

    for name, r in results.items():
        if "bits_per_dim" not in r:
            continue
        bits_per_dim = r["bits_per_dim"]
        total_bits = n_vectors * dim * bits_per_dim
        total_gb = total_bits / 8 / 1e9
        monthly_cost = total_gb * s3_price_per_gb

        print(f"\n  {name}:")
        print(f"    Storage: {total_gb:,.0f} GB")
        print(f"    Monthly cost: ${monthly_cost:,.0f}")

    # Savings comparison
    if "FP16" in results and "TQ-3bit" in results:
        fp16_gb = n_vectors * dim * 16 / 8 / 1e9
        tq3_bits = results["TQ-3bit"]["bits_per_dim"]
        tq3_gb = n_vectors * dim * tq3_bits / 8 / 1e9
        savings_gb = fp16_gb - tq3_gb
        savings_cost = savings_gb * s3_price_per_gb * 12  # annual

        print(f"\n  TurboQuant 3-bit vs FP16:")
        print(f"    Storage saved: {savings_gb:,.0f} GB ({savings_gb/fp16_gb:.0%} reduction)")
        print(f"    Annual savings: ${savings_cost:,.0f}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    results = run_benchmarks(data_dir=data_dir)
    notion_scale_projection(results, data_dir=data_dir)
