import json
import sys
from datetime import date
from pathlib import Path

import llm
from build_chunks import MAX_CHARS
from embed_chunks import load_index
from embedders import Embedder
from evaluate import POOL, RESULTS_DIR, TOP_K, first_hit_rank, print_heldout, print_report, summarize
from questions import load_questions
from rewrite import VARIANT, get_rewrite, search_with_rewrite

MODEL_KEY = "bge-base"

if __name__ == "__main__":
    run_name = sys.argv[1] if len(sys.argv) > 1 else "v2_bge_rewrite"
    chunks, embeddings = load_index(MODEL_KEY, include_recitals=False)
    embedder = Embedder(MODEL_KEY)

    details = []
    questions = load_questions()
    for i, q in enumerate(questions, start=1):
        rewritten = get_rewrite(q["question"])
        pool, original_top = search_with_rewrite(q["question"], rewritten, embedder, chunks, embeddings, POOL)
        results = pool[:TOP_K]
        has_answer = bool(q["expected"])
        details.append({
            "id": q["id"], "type": q["type"], "split": q["split"], "question": q["question"],
            "expected": q["expected"], "rewrite": rewritten,
            "rank": first_hit_rank(results, q["expected"]) if has_answer else None,
            "pool_rank": first_hit_rank(pool, q["expected"]) if has_answer else None,
            "top_score": round(original_top, 3),
            "retrieved": [{"label": c["label"], "score": round(s, 3)} for c, s in results],
        })
        print(f"  {i}/{len(questions)}  {q['id']}", flush=True)

    dev = [d for d in details if d["split"] != "heldout"]
    held = [d for d in details if d["split"] == "heldout"]
    dev_summary = summarize(dev)
    held_summary = summarize(held) if held else None
    print_report(run_name, dev_summary, dev)
    if held_summary:
        print_heldout(held_summary, held, False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    chunk_config_path = Path("data/processed/chunks_config.json")
    result = {
        "run": run_name, "date": str(date.today()),
        "config": {
            "model": embedder.name, "reranker": None, "max_chars": MAX_CHARS, "top_k": TOP_K, "pool": POOL,
            "chunks": len(chunks), "recitals_excluded": True, "query_rewriting": llm.MODEL,
            "rewrite_prompt": VARIANT,
            "chunking": json.loads(chunk_config_path.read_text(encoding="utf-8")) if chunk_config_path.exists() else None,
        },
        "summary": dev_summary, "heldout_summary": held_summary,
        "details": dev, "heldout_details": "hidden",
    }
    (RESULTS_DIR / f"{run_name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved:", RESULTS_DIR / f"{run_name}.json")