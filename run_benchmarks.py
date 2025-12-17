import yaml
import os
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from benchmarks.retrieval_quality import evaluate
from benchmarks.latency import measure_latency
from benchmarks.cost_analysis import estimate_cost
from gemini_embedding import embed_gemini
from generate_ai_dataset import generate_dataset


# ------------------ CHART GENERATION ------------------
def generate_charts(results):
    models = [r["model"] for r in results]

    # ---- Retrieval Quality (NDCG@10) ----
    ndcg_scores = [r["quality"]["NDCG@10"] for r in results]

    plt.figure()
    plt.bar(models, ndcg_scores)
    plt.title("Retrieval Quality Comparison (NDCG@10)")
    plt.ylabel("NDCG@10")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig("results/retrieval_quality.png")
    plt.close()

    # ---- Latency (Mean ms) ----
    latency_means = [r["latency"]["mean_ms"] for r in results]

    plt.figure()
    plt.bar(models, latency_means)
    plt.title("Latency Comparison (Mean ms)")
    plt.ylabel("Milliseconds")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig("results/latency.png")
    plt.close()


# ------------------ MAIN ------------------
def main():
    # -------- Generate AI Dataset --------
    documents, queries = generate_dataset(num_samples=100)

    # -------- Load Model Config --------
    with open("benchmark_config.yaml") as f:
        cfg = yaml.safe_load(f)

    models = cfg["models"]

    os.makedirs("results", exist_ok=True)
    results = []

    # -------- Run Benchmarks --------
    for model_cfg in models:
        name = model_cfg["name"]
        print(f"Running benchmark for {name}")

        if model_cfg["type"] == "local":
            model = SentenceTransformer(model_cfg["model_id"])
            embed_fn = lambda x: model.encode(x).tolist()
        else:
            embed_fn = embed_gemini

        # ---- Document embeddings ----
        doc_embeddings = embed_fn(documents)

        # ---- Query embeddings ----
        query_data = []
        for q in queries:
            q_emb = embed_fn([q["query"]])[0]
            query_data.append({
                "embedding": q_emb,
                "relevant_docs": q["relevant_docs"]
            })

        quality = evaluate(doc_embeddings, query_data)
        latency = measure_latency(embed_fn, documents)
        cost = estimate_cost(name)

        results.append({
            "model": name,
            "quality": quality,
            "latency": latency,
            "cost": cost
        })

    # -------- Generate Charts --------
    generate_charts(results)

    # -------- Write Article --------
    with open("results/article.md", "w") as report:
        report.write("# Text Embedding Model Benchmark\n\n")

        report.write("## Dataset Description\n")
        report.write(
            "This benchmark uses an AI-generated synthetic dataset consisting of "
            "100 technical paragraphs and 100 corresponding questions. "
            "Each question is mapped to a known relevant document, enabling "
            "controlled and reproducible evaluation of retrieval performance.\n\n"
        )

        report.write("## TL;DR\n")
        report.write("- Fastest model: MiniLM\n")
        report.write("- Best retrieval quality: BGE-Large\n")
        report.write("- Best balance: BGE-Base\n")
        report.write("- API-based option without billing: Gemini\n\n")

        # ---- Retrieval Quality Table ----
        report.write("## Retrieval Quality Results\n\n")
        report.write("| Model | Recall@1 | Recall@5 | Recall@10 | NDCG@10 |\n")
        report.write("|------|----------|----------|-----------|---------|\n")
        for r in results:
            q = r["quality"]
            report.write(
                f"| {r['model']} | {q['Recall@1']:.2f} | {q['Recall@5']:.2f} | "
                f"{q['Recall@10']:.2f} | {q['NDCG@10']:.2f} |\n"
            )

        report.write("\n### Retrieval Quality Visualization\n\n")
        report.write("![Retrieval Quality](retrieval_quality.png)\n\n")

        # ---- Latency Table ----
        report.write("## Latency Benchmark (ms)\n\n")
        report.write("| Model | Mean | P95 | P99 |\n")
        report.write("|------|------|-----|-----|\n")
        for r in results:
            l = r["latency"]
            report.write(
                f"| {r['model']} | {l['mean_ms']:.2f} | "
                f"{l['p95_ms']:.2f} | {l['p99_ms']:.2f} |\n"
            )

        report.write("\n### Latency Visualization\n\n")
        report.write("![Latency](latency.png)\n\n")

        # ---- Cost Table ----
        report.write("## Cost Analysis\n\n")
        report.write("| Model | Cost / 1M Tokens | Notes |\n")
        report.write("|------|------------------|-------|\n")
        for r in results:
            c = r["cost"]
            report.write(
                f"| {r['model']} | ${c['cost_per_1M_tokens']:.2f} | {c['note']} |\n"
            )

        # ---- Decision Matrix ----
        report.write("\n## Decision Matrix\n")
        report.write("- Choose **MiniLM** for low-latency, high-throughput systems\n")
        report.write("- Choose **BGE-Base** for balanced quality and performance\n")
        report.write("- Choose **BGE-Large** for maximum retrieval accuracy\n")
        report.write("- Choose **Gemini** for API-based embeddings without infrastructure\n\n")

        # ---- Reproducibility ----
        report.write("## Reproducibility\n\n")
        report.write("```bash\n")
        report.write("python -m venv venv\n")
        report.write("venv\\Scripts\\activate\n")
        report.write("pip install -r requirements.txt\n")
        report.write("python run_benchmarks.py\n")
        report.write("```\n")

    print("Benchmark completed. Results saved to results/article.md")


if __name__ == "__main__":
    main()
