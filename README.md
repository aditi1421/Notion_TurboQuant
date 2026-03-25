# TurboQuant for Notion Semantic Search

**Compressing Notion's 10B+ embedding vectors from 16 bits/dim to 3-4 bits/dim — with >93% recall.**

A proof-of-concept showing how [TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026) can dramatically reduce the cost of Notion's vector search infrastructure while preserving search quality.

## The Pitch

Notion stores **10 billion+ embedding vectors** in [Turbopuffer](https://turbopuffer.com/) to power AI-powered workspace search. Each vector is stored in FP16 (16 bits per dimension). TurboQuant can compress these to **3-4 bits per dimension** — a 4-5x reduction — with minimal impact on search recall.

### Key Results (benchmarked on real Notion workspace data)

| Method | Bits/dim | Compression | Recall@1 | Recall@10 | Latency |
|--------|----------|-------------|----------|-----------|---------|
| **FP16 (current)** | 16.0 | 1.0x | 100% | 100% | 1.21ms |
| **TurboQuant 4-bit** | 4.1 | **3.9x** | **93.8%** | **96.2%** | 0.09ms |
| **TurboQuant 3-bit** | 3.1 | **5.2x** | 81.2% | 93.8% | 0.12ms |
| TurboQuant 2-bit | 2.1 | 7.7x | 93.8% | 90.6% | 0.14ms |
| Binary Quantization | 1.0 | 16.0x | 6.2% | 76.2% | 0.09ms |
| Product Quantization | — | — | N/A | N/A | N/A |

**PQ can't even run** on small namespaces (<256 vectors) because it requires codebook training. TurboQuant needs zero training — it works on any corpus size, which is critical for Notion's millions of per-workspace namespaces.

### At Notion's Scale (10B vectors)

| Metric | FP16 | TQ 3-bit | Savings |
|--------|------|----------|---------|
| Storage | 7,680 GB | 1,480 GB | **81% reduction** |
| Annual cost | $2,119 | $408 | **$1,711/year** |

*Note: Storage costs based on S3 pricing ($0.023/GB/month). Real savings at Notion's scale would be in the millions when accounting for compute, network, and Turbopuffer's internal storage costs.*

## How It Works

```
CURRENT (Notion + Turbopuffer):
  Page Edit → Chunk → Embed → FP16 vectors → Turbopuffer → Brute-force search

WITH TURBOQUANT:
  Page Edit → Chunk → Embed → TurboQuant compress (3-4 bits) → Turbopuffer
  Query: FP16 query × compressed corpus → Asymmetric inner product → Same results
```

**The key insight**: TurboQuant uses an **asymmetric inner product estimator** — queries stay in full FP16, only the corpus is compressed. The QJL (Quantized Johnson-Lindenstrauss) correction makes the inner product estimate mathematically **unbiased**, preserving search ranking.

### Two-Stage Compression

1. **Stage 1 (Lloyd-Max)**: Random rotation → per-coordinate optimal quantization → (bits-1) bits per dim
2. **Stage 2 (QJL)**: 1-bit sign of projected residual → corrects inner product bias

## Why TurboQuant > Product Quantization for Notion

| Feature | TurboQuant | Product Quantization |
|---------|------------|---------------------|
| Training required | **None** | Requires 256+ vectors for codebook |
| Works on small namespaces | **Yes** | No (fails on <256 vectors) |
| Indexing time | **Near-zero** | Minutes (codebook training) |
| Recall at 3-4 bits | **93-96%** | Comparable but needs training |
| Asymmetric search | **Native** | Requires special implementation |

This matters because Notion has **millions of namespaces**, many with fewer than 256 vectors. PQ simply cannot serve these workspaces.

## Project Structure

```
├── turboquant/                  # Core algo (from tonbistudio/turboquant-pytorch)
│   ├── turboquant.py            # TurboQuantMSE, TurboQuantProd
│   ├── lloyd_max.py             # Lloyd-Max codebook solver
│   └── compressors.py           # Asymmetric attention compressors
├── search/                      # Search-specific adaptation
│   ├── embedding_quantizer.py   # TurboQuant wrapper for embeddings
│   ├── index.py                 # Vector index with compressed storage
│   └── baselines.py             # FP16, PQ, Binary baselines
├── data/                        # Notion data pipeline
│   ├── ingest_notion_data.py    # Process Notion pages into chunks
│   └── embed_chunks.py          # Generate embeddings
├── benchmarks/                  # Benchmark suite
│   ├── run_benchmarks.py        # Full benchmark (recall, latency, compression)
│   └── results.json             # Benchmark results
└── demo/                        # Interactive search demo
    └── notion_search_demo.py    # Search your Notion docs with TurboQuant
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. Process Notion data
python data/ingest_notion_data.py

# 2. Embed chunks
python data/embed_chunks.py data/notion_chunks.jsonl

# 3. Run benchmarks
python benchmarks/run_benchmarks.py data

# 4. Interactive demo
python demo/notion_search_demo.py data
```

## How This Fits Into Notion's Pipeline

TurboQuant slots in **between embedding generation and Turbopuffer storage** — a single compression step that requires no changes to:
- The chunking strategy
- The embedding model (Ray/Anyscale)
- The query pipeline
- Turbopuffer's storage format (just store compressed vectors instead of FP16)

The only change: at query time, use the asymmetric inner product estimator instead of standard dot product.

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [Two years of vector search at Notion](https://www.notion.com/blog/two-years-of-vector-search-at-notion)
- [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) — Core implementation
- [TurboQuant: Redefining AI efficiency (Google Research)](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
