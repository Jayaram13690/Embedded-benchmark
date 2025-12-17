# Text Embedding Model Benchmark

## TL;DR
- Fastest model: MiniLM
- Best retrieval quality: BGE-Large
- Best balance: BGE-Base
- API-based option without billing: Gemini

## Retrieval Quality Results

| Model | Recall@1 | Recall@5 | Recall@10 | NDCG@10 |
|------|----------|----------|-----------|---------|
| MiniLM | 1.00 | 1.00 | 1.00 | 1.00 |
| BGE-Base | 1.00 | 1.00 | 1.00 | 1.00 |
| BGE-Large | 1.00 | 1.00 | 1.00 | 1.00 |
| Gemini | 1.00 | 1.00 | 1.00 | 1.00 |

## Latency Benchmark (ms)

| Model | Mean | P95 | P99 |
|------|------|-----|-----|
| MiniLM | 13.62 | 22.15 | 23.44 |
| BGE-Base | 83.74 | 88.55 | 89.36 |
| BGE-Large | 333.53 | 393.04 | 396.73 |
| Gemini | 1108.62 | 1211.85 | 1235.70 |

## Cost Analysis

| Model | Cost / 1M Tokens | Notes |
|------|------------------|-------|
| MiniLM | $0.00 | Open-source, self-hosted (CPU/GPU cost excluded) |
| BGE-Base | $0.00 | Open-source, self-hosted (CPU/GPU cost excluded) |
| BGE-Large | $0.00 | Open-source, self-hosted (CPU/GPU cost excluded) |
| Gemini | $0.00 | Free tier Gemini API (rate-limited) |

## Decision Matrix
- Choose **MiniLM** for low latency systems
- Choose **BGE-Base** for balanced production use
- Choose **BGE-Large** for best semantic accuracy
- Choose **Gemini** if API-based embeddings are required

## Reproducibility

bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run_benchmarks.py

