import os
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def embed_gemini(texts):
    embeddings = []

    for text in texts:
        resp = client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )

        # Gemini returns a list of embeddings
        embeddings.append(resp.embeddings[0].values)

    return embeddings
