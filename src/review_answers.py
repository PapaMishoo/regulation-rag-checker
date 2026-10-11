import json
import sys
from pathlib import Path

from answer import RETRIEVAL_MODEL, TOP_K
from embed_chunks import load_index
from embedders import Embedder
from search import search

run = sys.argv[1]
ids = sys.argv[2:]
data = json.loads((Path("eval/answer_results") / f"{run}.json").read_text(encoding="utf-8"))
rows = {r["id"]: r for r in data["details"]}

chunks, embeddings = load_index(RETRIEVAL_MODEL, include_recitals=False)
embedder = Embedder(RETRIEVAL_MODEL)

for qid in ids:
    r = rows.get(qid)
    if r is None:
        print(f"\n{qid}: not in the dev results (held-out questions are hidden)")
        continue
    print("=" * 70)
    print(f"[{qid}] {r['question']}")
    print("Status:", r["status"])
    print("\nANSWER:", r["answer"])
    print("CITED:", ", ".join(r["citations"]) or "-", "| INVALID:", ", ".join(r["bad_citations"]) or "-")
    results = search(r["question"], embedder, chunks, embeddings, top_k=TOP_K)
    cited = set(r["citations"])
    if not cited:
        print("Passages it saw:", ", ".join(c["label"] for c, _ in results))
    else:
        print("\nTHE PASSAGES IT WAS GIVEN THAT IT CITED:")
        for chunk, _score in results:
            if chunk["label"] in cited:
                print(f"\n--- {chunk['id']} ---")
                print(chunk["text"])