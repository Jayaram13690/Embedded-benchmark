# Text Embedding Benchmark

## Overview

This project benchmarks multiple **text embedding models** to compare their performance on **semantic retrieval tasks**. The benchmark evaluates models across **retrieval quality**, **latency**, and **cost / total cost of ownership (TCO)**.

The goal is to enable **data-driven decision making** when selecting an embedding model, for production use cases such as:
- Semantic search
- Question answering systems
- RAG (Retrieval Augmented Generation)
- Knowledge base retrieval

## Problem Statement
Text embedding models differ significantly in:
- Retrieval accuracy
- Inference latency
- Deployment and operational cost

Choosing an unsuitable model can lead to:
- Poor search relevance
- High response latency
- Increased infrastructure or API costs

This project provides a **practical, reproducible benchmark** to compare embedding models using **real execution and measurable metrics**.

## Models Evaluated

### Local (Open-Source Models)
- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-base-en-v1.5`
- `BAAI/bge-large-en-v1.5`

### API-Based Model
- **Gemini Embeddings** (Free Tier)

Local models are self-hosted, while Gemini represents an API-based embedding option without billing requirements during evaluation.

## Benchmarks Performed

### 1. Retrieval Quality
Measures how effectively embeddings retrieve relevant documents.
**Metrics**
- Recall@1
- Recall@5
- Recall@10
- NDCG@10

### 2. Latency
Measures the time required to generate embeddings.
**Metrics**
- Mean latency (ms)
- P95 latency (ms)
- P99 latency (ms)

### 3. Cost & TCO
Evaluates the economic trade-offs between:
- API-based usage
- Self-hosted infrastructure (CPU/GPU, memory, maintenance)

## Dataset Used

### AI-Generated Synthetic Dataset
The benchmark uses an **AI-generated synthetic dataset** consisting of:
- 100 technical paragraphs (documents)
- 100 corresponding questions (queries)
- Explicit relevance mapping (each query maps to a known relevant document)

This dataset simulates a **knowledge-base retrieval scenario** and enables controlled and reproducible evaluation of embedding models.

> **Note:**  
> The dataset is synthetic and clearly documented as such.  
> The benchmarking pipeline supports swapping in real-world datasets
> (e.g., MS MARCO, NFCorpus) without changing evaluation logic.

## Project Structure
```
├── benchmarks/
│ ├── retrieval_quality.py # Recall & NDCG metrics
│ ├── latency.py           # Latency measurement
│ └── cost_analysis.py     # Cost & TCO estimation
├── generate_ai_dataset.py # AI-generated dataset
├── gemini_embedding.py    # Gemini embedding client
├── run_benchmarks.py      # Benchmark orchestrator
├── benchmark_config.yaml  # Model configuration
├── requirements.txt
├── results/
│ ├── article.md            # Generated benchmark report
│ ├── retrieval_quality.png # Retrieval quality visualization
│ └── latency.png           # Latency comparison visualization
└── README.md
```

## Workflow

1. Generate dataset (documents and queries)
2. Generate embeddings for documents and queries
3. Perform similarity-based retrieval
4. Compute retrieval metrics
5. Measure embedding latency
6. Analyze cost and TCO
7. Generate tables, charts, and a final report

## How to Run

### 1. Create Virtual Environment
bash
```python -m venv venv```
```venv\Scripts\activate```
```pip install -r requirements.txt```
```python run_benchmarks.py```

The report includes:
1. TL;DR summary
2. Retrieval quality tables
3. Latency tables
4. Cost analysis
5. Charts for quality and latency
6. Decision matrix
7. Reproduction steps

## Expected Insights
- ***MiniLM*** → Fastest model, suitable for low-latency, high-throughput systems
- ***BGE-Base*** → Balanced performance and accuracy
- ***BGE-Large*** → Best retrieval quality
- ***Gemini*** → API-based solution without infrastructure setup

## Decision Matrix

| Use Case                      | Recommended Model |
| ----------------------------- | ----------------- |
| Low latency / high throughput | MiniLM            |
| Balanced production workloads | BGE-Base          |
| Maximum retrieval accuracy    | BGE-Large         |
| API-only solution             | Gemini            |


### Latency Visualization

![Latency](latency.png)


## Notes on Cost
All evaluated models show $0.00 cost during this benchmark because:
- Local models are open-source and self-hosted
- Gemini embeddings are accessed using a free tier
In production environments, infrastructure or API usage costs would apply and should be evaluated separately.

## Reproducibility
- No hardcoded results
- All metrics computed from real executions
- No mock or placeholder data
- Fully reproducible on another machine.

## Conclusion
This project describes complete end-to-end benchmarking of text embedding models, Practical evaluation of retrieval quality and latency, Clear trade-off analysis for production decision making.
The benchmark is designed to be ***transparent, extensible, and production-oriented***.





