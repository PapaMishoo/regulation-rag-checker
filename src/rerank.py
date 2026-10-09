import numpy as np
from sentence_transformers import CrossEncoder

from embed_chunks import chunk_to_text

RERANKER_NAME = "BAAI/bge-reranker-base"


def load_reranker():
    return CrossEncoder(RERANKER_NAME, max_length=512)


def rerank(query, candidates, reranker):
    """Re-order search candidates by reading the question and each passage together."""
    pairs = [(query, chunk_to_text(chunk)) for chunk, _score in candidates]
    scores = reranker.predict(pairs)
    order = np.argsort(scores)[::-1]
    return [(candidates[i][0], float(scores[i])) for i in order]