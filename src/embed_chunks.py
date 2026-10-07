import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
PROCESSED_DIR = Path("data/processed")
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"


def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chunk_to_text(chunk):
    # The label goes in front so the model knows which law and article this is
    return f"{chunk['label']}. {chunk['text']}"


if __name__ == "__main__":
    chunks = load_chunks()
    model = SentenceTransformer(MODEL_NAME)
    texts = [chunk_to_text(c) for c in chunks]

    # How many chunks are longer than the model can read?
    limit = model.max_seq_length
    token_counts = [len(model.tokenizer(t)["input_ids"]) for t in texts]
    too_long = sum(1 for n in token_counts if n > limit)
    print("Model:", MODEL_NAME)
    print("Model reads up to", limit, "tokens per chunk")
    print(f"Chunks longer than that (the end gets ignored): {too_long} of {len(texts)}")

    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    print("Saved embeddings with shape:", embeddings.shape)