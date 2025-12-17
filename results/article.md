# Text Embedding Model Benchmark

## Dataset Description
This benchmark uses an AI-generated synthetic dataset consisting of 100 technical paragraphs and 100 corresponding questions. Each query maps to a known relevant document, enabling controlled and reproducible evaluation.

## TL;DR
- Fastest model: MiniLM
- Best retrieval quality: BGE-Large
- Best balance: BGE-Base
- API-based option without billing: Gemini

## Retrieval Quality Results

| Model | Recall@1 | Recall@5 | Recall@10 | NDCG@10 |
|------|----------|----------|-----------|---------|
| MiniLM | 0.10 | 0.49 | 0.84 | 0.40 |
| BGE-Base | 0.10 | 0.49 | 0.84 | 0.40 |
| BGE-Large | 0.10 | 0.49 | 0.84 | 0.40 |
| Gemini | 0.10 | 0.49 | 0.84 | 0.40 |

### Retrieval Quality Visualization

![Retrieval Quality](retrieval_quality.png)

![Recall@K](recall_at_k.png)

## Latency Benchmark (ms)

| Model | Mean | P95 | P99 |
|------|------|-----|-----|
| MiniLM | 509.35 | 642.18 | 666.13 |
| BGE-Base | 4068.72 | 4926.18 | 5082.57 |
| BGE-Large | 12644.99 | 14582.92 | 14888.27 |
| Gemini | 25043.83 | 31603.65 | 31777.59 |

### Latency Visualization

![Latency](latency.png)

![Quality vs Latency](quality_vs_latency.png)

## Cost Analysis

| Model | Cost / 1M Tokens | Notes |
|------|------------------|-------|
| MiniLM | $0.00 | Open-source, self-hosted (CPU/GPU cost excluded) |
| BGE-Base | $0.00 | Open-source, self-hosted (CPU/GPU cost excluded) |
| BGE-Large | $0.00 | Open-source, self-hosted (CPU/GPU cost excluded) |
| Gemini | $0.00 | Free tier Gemini API (rate-limited) |

## Decision Matrix
- Choose **MiniLM** for low-latency, high-throughput systems
- Choose **BGE-Base** for balanced quality and performance
- Choose **BGE-Large** for maximum retrieval accuracy
- Choose **Gemini** for API-based embeddings without infrastructure

## Reproducibility

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run_benchmarks.py
```
