from sentence_transformers import SentenceTransformer

# Each model has a short name we use on the command line
MODELS = {
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "query_prefix": "",
    },
    "bge-base": {
        "name": "BAAI/bge-base-en-v1.5",
        # BGE models expect this sentence in front of search questions
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
}
DEFAULT_MODEL = "minilm"


class Embedder:
    """Wraps an embedding model so every script uses it the same way."""

    def __init__(self, key):
        if key not in MODELS:
            raise SystemExit(f"Unknown model '{key}'. Choose from: {', '.join(MODELS)}")
        self.key = key
        self.name = MODELS[key]["name"]
        self.query_prefix = MODELS[key]["query_prefix"]
        self.model = SentenceTransformer(self.name)

    def encode_docs(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    def encode_query(self, query):
        return self.model.encode(self.query_prefix + query, normalize_embeddings=True)