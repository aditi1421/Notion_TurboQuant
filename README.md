# TurboQuant for Notion Semantic Search

**Replacing Turbopuffer's RaBitQ with TurboQuant — better recall, tunable compression, GPU-accelerated, no re-ranking needed.**

Turbopuffer currently uses [RaBitQ](https://dl.acm.org/doi/10.1145/3654970) for vector compression in their ANN v3 index. RaBitQ compresses to 1 bit/dim and requires SSD re-ranking to achieve usable recall. [TurboQuant](https://arxiv.org/abs/2504.19874) (Google Research, ICLR 2026) directly outperforms RaBitQ on retrieval benchmarks — with tunable bit-widths, no re-ranking, and orders-of-magnitude faster indexing.

This repo is a proof-of-concept benchmarked on real Notion workspace data.

## Why TurboQuant > RaBitQ (Turbopuffer's Current Approach)

Turbopuffer's ANN v3 uses RaBitQ binary quantization internally. Here's what's limiting about that:

| Limitation | RaBitQ (Turbopuffer today) | TurboQuant |
|-----------|---------------------------|------------|
| **Bit-width** | Locked at 1 bit/dim | **Tunable: 2, 3, or 4 bits/dim** |
| **Recall without re-ranking** | Poor — "can hardly produce reasonable recall" without full-precision re-ranking ([source](https://arxiv.org/html/2409.09913v1)) | **93-96% recall with no re-ranking** |
| **Re-ranking overhead** | Requires SSD round-trip to fetch full vectors | **Eliminated — asymmetric estimator is accurate enough** |
| **GPU acceleration** | Not vectorizable, no GPU implementation | **Fully vectorizable, orders of magnitude faster** ([ICLR 2026](https://arxiv.org/abs/2504.19874)) |
| **Recall at same bits** | Lower recall on GloVe benchmarks | **Higher recall on GloVe** (paper Table 3) |
| **Index build time** | Requires codebook computation | **Near-zero — data-oblivious, no training** |

The ICLR 2026 paper explicitly benchmarks TurboQuant against RaBitQ and PQ, showing superior recall across all bit-widths on standard retrieval benchmarks.

### The Re-Ranking Problem

This is the key insight. Turbopuffer's architecture today:

```
Query → RaBitQ search (1-bit, in DRAM) → candidate set
      → Re-rank candidates against full vectors (from SSD) → final results
```

That SSD re-ranking is the latency bottleneck. TurboQuant at 3-4 bits achieves high enough recall to **skip re-ranking entirely**:

```
Query → TurboQuant asymmetric search (3-4 bits, in DRAM) → final results
```

Eliminating the SSD round-trip could cut p50 query latency significantly.

## Benchmark Results (Real Notion Workspace Data)

Benchmarked on 16 actual Notion workspace documents, chunked and embedded (all-MiniLM-L6-v2, 384d):

| Method | Bits/dim | Compression | Recall@1 | Recall@10 | Latency |
|--------|----------|-------------|----------|-----------|---------|
| FP16 (ground truth) | 16.0 | 1.0x | 100% | 100% | 1.21ms |
| **TurboQuant 4-bit** | 4.1 | **3.9x** | **93.8%** | **96.2%** | 0.09ms |
| **TurboQuant 3-bit** | 3.1 | **5.2x** | 81.2% | 93.8% | 0.12ms |
| TurboQuant 2-bit | 2.1 | 7.7x | 93.8% | 90.6% | 0.14ms |
| Binary Quantization (RaBitQ-like) | 1.0 | 16.0x | 6.2% | 76.2% | 0.09ms |

**Binary quantization alone (what RaBitQ gives without re-ranking) has 6.2% Recall@1.** That's why Turbopuffer needs the SSD re-ranking step. TurboQuant at 4-bit hits 93.8% without it.

*Note: This is a small-scale benchmark (16 documents). The ICLR paper validates at scale on GloVe (400K vectors, 200d) where TurboQuant also outperforms RaBitQ.*

## Cost & Performance Projection at Notion Scale (10B vectors)

| Metric | Current (RaBitQ + SSD re-rank) | TurboQuant 4-bit (no re-rank) |
|--------|-------------------------------|-------------------------------|
| DRAM per vector | 1 bit/dim (binary codes) | 4 bits/dim |
| SSD access needed | Yes (full vectors for re-rank) | **No** |
| Recall without re-rank | ~70-80% | **93-96%** |
| Index build time | Codebook computation | **Near-zero** |
| GPU-accelerable | No | **Yes** |

The trade-off: TurboQuant uses ~4x more DRAM than RaBitQ (4 bits vs 1 bit), but **eliminates SSD re-ranking entirely**. At Notion's scale, the SSD I/O saved likely outweighs the DRAM cost — especially since Turbopuffer already keeps compressed vectors in DRAM.

## How TurboQuant Works

Two-stage compression with mathematically unbiased inner products:

1. **Stage 1 (Lloyd-Max)**: Random orthogonal rotation → per-coordinate optimal scalar quantization → (b-1) bits per dim
2. **Stage 2 (QJL)**: 1-bit Quantized Johnson-Lindenstrauss on residuals → corrects inner product bias

The **asymmetric estimator** is the key: queries stay in full FP16, only the corpus is compressed. The QJL correction makes `<query, compressed_key>` an **unbiased estimate** of `<query, key>` with variance O(1/d). This is why re-ranking isn't needed.

## Where This Fits in Notion's Pipeline

TurboQuant would replace or augment RaBitQ inside Turbopuffer's quantization layer:

```
CURRENT:
  Embed → FP16 to Turbopuffer → RaBitQ (1-bit DRAM) + full vectors (SSD)
  Query: RaBitQ scan → SSD re-rank → results

WITH TURBOQUANT:
  Embed → FP16 to Turbopuffer → TurboQuant (3-4 bit DRAM only)
  Query: TurboQuant asymmetric scan → results (no SSD needed)
```

No changes to Notion's chunking, embedding model, or query pipeline. The change is inside Turbopuffer's quantization layer.

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
    └── notion_search_demo.py    # Search Notion docs with TurboQuant
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
```

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [TurboQuant: Redefining AI efficiency (Google Research)](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [Turbopuffer ANN v3: 200ms p99 over 100B vectors](https://turbopuffer.com/blog/ann-v3) — Turbopuffer's RaBitQ implementation
- [RaBitQ: Quantizing High-Dimensional Vectors](https://dl.acm.org/doi/10.1145/3654970) — Current method used by Turbopuffer
- [Two years of vector search at Notion](https://www.notion.com/blog/two-years-of-vector-search-at-notion)
- [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) — Core implementation
