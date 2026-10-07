import numpy as np
from sentence_transformers import SentenceTransformer

from embed_chunks import EMBEDDINGS_PATH, MODEL_NAME, load_chunks

TOP_K = 3  # how many results to show


def search(query, model, chunks, embeddings, top_k=TOP_K):
    query_vec = model.encode(query, normalize_embeddings=True)
    scores = embeddings @ query_vec            # similarity of the question to every chunk
    best = np.argsort(scores)[::-1][:top_k]    # positions of the highest scores
    return [(chunks[i], float(scores[i])) for i in best]


def show(results):
    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"\n{rank}. {chunk['label']}  (score {score:.2f})")
        print("   " + chunk["text"][:300] + "...")


if __name__ == "__main__":
    chunks = load_chunks()
    embeddings = np.load(EMBEDDINGS_PATH)
    if len(chunks) != len(embeddings):
        raise SystemExit("Chunks and embeddings don't match. Run: python src/embed_chunks.py")

    model = SentenceTransformer(MODEL_NAME)
    print("Type a question, or press Enter on an empty line to stop.")
    while True:
        query = input("\nQuestion: ").strip()
        if not query:
            break
        show(search(query, model, chunks, embeddings))