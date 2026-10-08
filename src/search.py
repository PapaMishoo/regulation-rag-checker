import sys

import numpy as np

from embed_chunks import load_index
from embedders import DEFAULT_MODEL, Embedder

TOP_K = 3  # how many results to show


def search(query, embedder, chunks, embeddings, top_k=TOP_K):
    query_vec = embedder.encode_query(query)
    scores = embeddings @ query_vec            # similarity of the question to every chunk
    best = np.argsort(scores)[::-1][:top_k]    # positions of the highest scores
    return [(chunks[i], float(scores[i])) for i in best]


def show(results):
    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"\n{rank}. {chunk['label']}  (score {score:.2f})")
        print("   " + chunk["text"][:300] + "...")


if __name__ == "__main__":
    model_key = DEFAULT_MODEL
    for arg in sys.argv[1:]:
        if arg.startswith("--model="):
            model_key = arg.split("=", 1)[1]

    # Recitals are left out of search (decided in Experiment 1)
    chunks, embeddings = load_index(model_key, include_recitals=False)
    embedder = Embedder(model_key)
    print("Type a question, or press Enter on an empty line to stop.")
    while True:
        query = input("\nQuestion: ").strip()
        if not query:
            break
        show(search(query, embedder, chunks, embeddings))