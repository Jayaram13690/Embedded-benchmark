import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recall_at_k(ranked, relevant, k):
    return int(any(i in ranked[:k] for i in relevant))

def ndcg_at_10(ranked, relevant):
    score = 0.0
    for i, idx in enumerate(ranked[:10]):
        if idx in relevant:
            score += 1 / np.log2(i + 2)
    return score

def evaluate(doc_embeddings, queries):
    recalls = {1: [], 5: [], 10: []}
    ndcgs = []

    docs = np.array(doc_embeddings)

    for q in queries:
        q_vec = np.array(q["embedding"]).reshape(1, -1)
        sims = cosine_similarity(q_vec, docs)[0]
        ranked = np.argsort(sims)[::-1]

        for k in recalls:
            recalls[k].append(recall_at_k(ranked, q["relevant_docs"], k))

        ndcgs.append(ndcg_at_10(ranked, q["relevant_docs"]))

    return {
        "Recall@1": float(np.mean(recalls[1])),
        "Recall@5": float(np.mean(recalls[5])),
        "Recall@10": float(np.mean(recalls[10])),
        "NDCG@10": float(np.mean(ndcgs))
    }
