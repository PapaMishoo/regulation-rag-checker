import json
import sys
from datetime import date
from pathlib import Path

from build_chunks import MAX_CHARS
from embed_chunks import load_index
from embedders import DEFAULT_MODEL, Embedder
from rerank import RERANKER_NAME, load_reranker, rerank
from search import search

QUESTIONS_PATH = Path("eval/questions.json")
RESULTS_DIR = Path("eval/results")
TOP_K = 5   # results we score
POOL = 20   # candidates the search hands to the re-ranker
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]


def first_hit_rank(results, expected):
    """Position (1 = best) of the first result that is an expected article, else None."""
    for rank, (chunk, _score) in enumerate(results, start=1):
        if chunk["label"] in expected:
            return rank
    return None


def run_questions(embedder, chunks, embeddings, reranker=None):
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    details = []
    for q in questions:
        pool = search(q["question"], embedder, chunks, embeddings, top_k=POOL)
        if reranker is not None:
            results = rerank(q["question"], pool, reranker)[:TOP_K]
        else:
            results = pool[:TOP_K]
        has_answer = bool(q["expected"])
        details.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "expected": q["expected"],
            "rank": first_hit_rank(results, q["expected"]) if has_answer else None,
            "pool_rank": first_hit_rank(pool, q["expected"]) if has_answer else None,
            "top_score": round(pool[0][1], 3),  # best search similarity, used for the cutoff test
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
        "pool_hit": sum(1 for d in scored if d["pool_rank"]) / len(scored),
        "by_type": by_type,
        "thresholds": thresholds,
    }


def print_report(run_name, summary, details):
    print("=" * 60)
    print("Run:", run_name)
    print(f"Questions with a right answer: {summary['n_scored']}   Out of scope: {summary['n_out_of_scope']}")
    print(f"Hit@1: {summary['hit_at_1']:.0%}   Hit@3: {summary['hit_at_3']:.0%}   "
          f"Hit@5: {summary['hit_at_5']:.0%}   MRR: {summary['mrr']:.2f}")
    print(f"Right article somewhere in the top {POOL} search candidates: {summary['pool_hit']:.0%}")

    print("\nHit@3 by question type:")
    for t, v in summary["by_type"].items():
        print(f"  {t:<8} {v['hit_at_3']:.0%}  ({v['n']} questions)")

    print("\nScore cutoff test (what if we refuse when the best search score is below the cutoff?):")
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
            print(f"       search rank of the right article: {d['pool_rank'] or 'not in top ' + str(POOL)}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    run_name = args[0] if args else "baseline"
    no_recitals = "--no-recitals" in flags
    use_rerank = "--rerank" in flags
    model_key = DEFAULT_MODEL
    for f in flags:
        if f.startswith("--model="):
            model_key = f.split("=", 1)[1]

    chunks, embeddings = load_index(model_key, include_recitals=not no_recitals)
    embedder = Embedder(model_key)
    reranker = load_reranker() if use_rerank else None

    details = run_questions(embedder, chunks, embeddings, reranker)
    summary = summarize(details)
    print_report(run_name, summary, details)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "run": run_name,
        "date": str(date.today()),
        "config": {
            "model": embedder.name,
            "reranker": RERANKER_NAME if use_rerank else None,
            "max_chars": MAX_CHARS,
            "top_k": TOP_K,
            "pool": POOL,
            "chunks": len(chunks),
            "recitals_excluded": no_recitals,
        },
        "summary": summary,
        "details": details,
    }
    out_path = RESULTS_DIR / f"{run_name}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved:", out_path)