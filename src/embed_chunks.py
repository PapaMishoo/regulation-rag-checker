import json
import sys
from pathlib import Path

import numpy as np

from embedders import DEFAULT_MODEL, Embedder

PROCESSED_DIR = Path("data/processed")
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"


def embeddings_path(model_key):
    return PROCESSED_DIR / f"embeddings_{model_key}.npy"


def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def chunk_to_text(chunk):
    # The label goes in front so the model knows which law and article this is
    return f"{chunk['label']}. {chunk['text']}"


def load_index(model_key, include_recitals=True):
    """Load chunks and their embeddings, optionally leaving out the recitals."""
    chunks = load_chunks()
    embeddings = np.load(embeddings_path(model_key))
    if len(chunks) != len(embeddings):
        raise SystemExit(f"Chunks and embeddings don't match. Run: python src/embed_chunks.py {model_key}")
    if not include_recitals:
        keep = [i for i, c in enumerate(chunks) if not c["label"].endswith("Preamble")]
        chunks = [chunks[i] for i in keep]
        embeddings = embeddings[keep]
    return chunks, embeddings


if __name__ == "__main__":
    model_key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    chunks = load_chunks()
    embedder = Embedder(model_key)
    texts = [chunk_to_text(c) for c in chunks]

    # How many chunks are longer than the model can read?
    limit = embedder.model.max_seq_length
    token_counts = [len(embedder.model.tokenizer(t)["input_ids"]) for t in texts]
    too_long = sum(1 for n in token_counts if n > limit)
    print("Model:", embedder.name)
    print("Model reads up to", limit, "tokens per chunk")
    print(f"Chunks longer than that (the end gets ignored): {too_long} of {len(texts)}")

    embeddings = embedder.encode_docs(texts)
    out_path = embeddings_path(model_key)
    np.save(out_path, embeddings)
    print("Saved embeddings with shape:", embeddings.shape, "to", out_path)