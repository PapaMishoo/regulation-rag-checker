import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from build_chunks import MAX_CHARS
from embed_chunks import EMBEDDINGS_PATH, MODEL_NAME, load_chunks
from search import search

QUESTIONS_PATH = Path("eval/questions.json")
RESULTS_DIR = Path("eval/results")
TOP_K = 5
THRESHOLDS = [0.2, 0.3, 0.4, 0.5]


def first_hit_rank(results, expected):
    """Position (1 = best) of the first result that is an expected article, else None."""
    for rank, (chunk, _score) in enumerate(results, start=1):
        if chunk["label"] in expected:
            return rank
    return None


def run_questions(model, chunks, embeddings):
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    details = []
    for q in questions:
        results = search(q["question"], model, chunks, embeddings, top_k=TOP_K)
        details.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "expected": q["expected"],
            "rank": first_hit_rank(results, q["expected"]) if q["expected"] else None,
            "top_score": round(results[0][1], 3),
            "retrieved": [{"label": c["label"], "score": round(s, 3)} for c, s in results],
        })
    return details


def summarize(details):
    scored = [d for d in details if d["expected"]]       # questions that have a right answer
    out_of_scope = [d for d in details if not d["expected"]]

    def hit_at(k, items):
        return sum(1 for d in items if d["rank"] and d["rank"] <= k) / len(items)

    by_type = {}
    for t in sorted({d["type"] for d in scored}):
        items = [d for d in scored if d["type"] == t]
        by_type[t] = {"n": len(items), "hit_at_3": hit_at(3, items)}

    thresholds = {}
    for t in THRESHOLDS:
        thresholds[str(t)] = {
            "real_questions_kept": sum(1 for d in scored if d["top_score"] >= t) / len(scored),
            "out_of_scope_refused": sum(1 for d in out_of_scope if d["top_score"] < t) / len(out_of_scope),
        }

    return {
        "n_scored": len(scored),
        "n_out_of_scope": len(out_of_scope),
        "hit_at_1": hit_at(1, scored),
        "hit_at_3": hit_at(3, scored),
        "hit_at_5": hit_at(5, scored),
        "mrr": sum(1 / d["rank"] for d in scored if d["rank"]) / len(scored),
        "by_type": by_type,
        "thresholds": thresholds,
    }


def print_report(run_name, summary, details):
    print("=" * 60)
    print("Run:", run_name)
    print(f"Questions with a right answer: {summary['n_scored']}   Out of scope: {summary['n_out_of_scope']}")
    print(f"Hit@1: {summary['hit_at_1']:.0%}   Hit@3: {summary['hit_at_3']:.0%}   "
          f"Hit@5: {summary['hit_at_5']:.0%}   MRR: {summary['mrr']:.2f}")

    print("\nHit@3 by question type:")
    for t, v in summary["by_type"].items():
        print(f"  {t:<8} {v['hit_at_3']:.0%}  ({v['n']} questions)")

    print("\nScore cutoff test (what if we refuse when the best score is below the cutoff?):")
    for t, v in summary["thresholds"].items():
        print(f"  cutoff {t}: real questions kept {v['real_questions_kept']:.0%}, "
              f"out-of-scope refused {v['out_of_scope_refused']:.0%}")

    print("\nMisses (right article not in the top 3):")
    for d in details:
        if d["expected"] and (d["rank"] is None or d["rank"] > 3):
            top = d["retrieved"][0]
            print(f"  [{d['id']}] {d['question']}")
            print(f"       wanted: {', '.join(d['expected'])}")
            print(f"       got #1: {top['label']} ({top['score']})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    run_name = args[0] if args else "baseline"
    no_recitals = "--no-recitals" in flags

    chunks = load_chunks()
    embeddings = np.load(EMBEDDINGS_PATH)

    if no_recitals:
        # Keep only articles and annexes; drop the recitals (labelled "Preamble")
        keep = [i for i, c in enumerate(chunks) if not c["label"].endswith("Preamble")]
        chunks = [chunks[i] for i in keep]
        embeddings = embeddings[keep]

    model = SentenceTransformer(MODEL_NAME)

    details = run_questions(model, chunks, embeddings)
    summary = summarize(details)
    print_report(run_name, summary, details)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "run": run_name,
        "date": str(date.today()),
        "config": {
            "model": MODEL_NAME,
            "max_chars": MAX_CHARS,
            "top_k": TOP_K,
            "chunks": len(chunks),
            "recitals_excluded": no_recitals,
        },
        "summary": summary,
        "details": details,
    }
    out_path = RESULTS_DIR / f"{run_name}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved:", out_path)