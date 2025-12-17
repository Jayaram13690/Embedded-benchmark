import random

TOPICS = [
    "machine learning",
    "deep learning",
    "natural language processing",
    "computer vision",
    "artificial intelligence",
    "neural networks",
    "transformers",
    "python programming",
    "data science",
    "embeddings"
]

def generate_paragraph(topic, idx):
    return (
        f"{topic.title()} is an important area in modern computing. "
        f"This paragraph {idx} explains core concepts of {topic}, "
        f"its applications, benefits, and challenges in real-world systems."
    )

def generate_question(topic):
    return f"What is {topic}?"

def generate_dataset(num_samples=100):
    documents = []
    queries = []

    for i in range(num_samples):
        topic = random.choice(TOPICS)

        doc = generate_paragraph(topic, i)
        documents.append(doc)

        queries.append({
            "query": generate_question(topic),
            "relevant_docs": [i]
        })

    return documents, queries


if __name__ == "__main__":
    docs, qs = generate_dataset()

    print("Documents:", len(docs))
    print("Queries:", len(qs))
    print("Sample document:", docs[0])
    print("Sample query:", qs[0])
