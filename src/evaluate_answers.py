import json
import sys
import time
from datetime import date
from pathlib import Path

import llm
from answer import RETRIEVAL_MODEL, answer_question
from embed_chunks import load_index
from embedders import Embedder
from questions import load_questions

RESULTS_DIR = Path("eval/answer_results")
CACHE_DIR = Path("data/processed")  # git-ignored; holds every row so a run can be resumed
PAUSE = 8  # seconds between model calls
REFUSED = {"refused_low_score", "refused_by_model"}
FAILED = {"api_error", "unreadable_reply", "quota_exhausted"}


def judge(q, out):
    expected = q["expected"]
    failed = out["status"] in FAILED
    row = {
        "id": q["id"], "type": q["type"], "split": q["split"], "question": q["question"],
        "expected": expected, "status": out["status"], "failed": failed, "answer": out["answer"],
        "citations": out["citations"], "bad_citations": out["bad_citations"],
        "sources": out["sources"], "top_score": out["top_score"],
        "seconds": out["seconds"], "tokens_in": out["input_tokens"], "tokens_out": out["output_tokens"],
    }
    if not failed:
        refused = out["status"] in REFUSED
        if expected:
            row["in_passages"] = any(e in out["sources"] for e in expected)
            row["cited_expected"] = any(c in expected for c in out["citations"])
            row["false_refusal"] = refused
        else:
            row["correctly_refused"] = refused
    return row


def share(items, key):
    return sum(1 for r in items if r.get(key)) / max(1, len(items))


def summarize(rows):
    ok = [r for r in rows if not r["failed"]]
    real = [r for r in ok if r["expected"]]
    off = [r for r in ok if not r["expected"]]
    answered = [r for r in real if not r["false_refusal"]]
    called = [r for r in ok if r["seconds"]]
    return {
        "n_real": len(real),
        "n_off_topic": len(off),
        "n_failed": len(rows) - len(ok),
        "false_refusal_rate": share(real, "false_refusal"),
        "right_article_in_passages": share(real, "in_passages"),
        "cited_expected_of_all_real": share(real, "cited_expected"),
        "cited_expected_of_answered": share(answered, "cited_expected"),
        "off_topic_refused": share(off, "correctly_refused"),
        "answers_with_invalid_citation": sum(1 for r in ok if r["bad_citations"]),
        "avg_seconds_per_call": round(sum(r["seconds"] for r in called) / max(1, len(called)), 1),
        "avg_tokens_in": round(sum(r["tokens_in"] for r in called) / max(1, len(called))),
    }


def print_summary(title, s, not_attempted=0):
    print("=" * 60)
    print(title)
    print(f"Scored: {s['n_real']} real + {s['n_off_topic']} off-topic questions."
          f"   Failed calls: {s['n_failed']}   Not attempted: {not_attempted}")
    print(f"Wrongly refused real questions:                 {s['false_refusal_rate']:.0%}")
    print(f"Right article among the 5 passages it was given: {s['right_article_in_passages']:.0%}")
    print(f"Cited the expected article (of all real):        {s['cited_expected_of_all_real']:.0%}")
    print(f"Cited the expected article (of those answered):  {s['cited_expected_of_answered']:.0%}")
    print(f"Off-topic questions refused:                     {s['off_topic_refused']:.0%}")
    print(f"Answers citing something it wasn't given:        {s['answers_with_invalid_citation']}")
    print(f"Average time per call: {s['avg_seconds_per_call']}s   Average input tokens: {s['avg_tokens_in']}")


def print_problems(rows):
    ok = [r for r in rows if not r["failed"]]
    print("\nRefused, but should have answered:")
    for r in ok:
        if r["expected"] and r["false_refusal"]:
            print(f"  [{r['id']}] {r['question']}  ({r['status']}, best score {r['top_score']})")
    print("\nAnswered, but should have refused:")
    for r in ok:
        if not r["expected"] and not r["correctly_refused"]:
            print(f"  [{r['id']}] {r['question']}  ({r['status']}, best score {r['top_score']})")
    print("\nAnswered without citing the expected article:")
    for r in ok:
        if r["expected"] and not r["false_refusal"] and not r["cited_expected"]:
            print(f"  [{r['id']}] cited: {', '.join(r['citations']) or 'nothing'}  | expected: {', '.join(r['expected'])}")
    print("\nInvalid citations:")
    for r in ok:
        if r["bad_citations"]:
            print(f"  [{r['id']}] {', '.join(r['bad_citations'])}")
    failed = [r for r in rows if r["failed"]]
    if failed:
        print(f"\nFailed calls: {len(failed)}. First error: {failed[0]['answer'][:200]}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    run_name = args[0] if args else "answers"
    reveal = "--reveal-heldout" in flags
    resume = "--resume" in flags
    limit = None
    for f in flags:
        if f.startswith("--limit="):
            limit = int(f.split("=", 1)[1])

    chunks, embeddings = load_index(RETRIEVAL_MODEL, include_recitals=False)
    embedder = Embedder(RETRIEVAL_MODEL)
    questions = load_questions()
    if limit:
        questions = questions[:limit]

    cache_path = CACHE_DIR / f"answer_cache_{run_name}.json"
    cache = {}
    if resume and cache_path.exists():
        cache = {r["id"]: r for r in json.loads(cache_path.read_text(encoding="utf-8"))}

    rows, stopped = [], False
    for i, q in enumerate(questions, start=1):
        old = cache.get(q["id"])
        if old and not old["failed"]:
            rows.append(old)          # already done in an earlier attempt
            continue
        if stopped:
            continue
        out = answer_question(q["question"], embedder, chunks, embeddings)
        row = judge(q, out)
        rows.append(row)
        cache[q["id"]] = row
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(list(cache.values()), ensure_ascii=False), encoding="utf-8")
        hidden = q["split"] == "heldout" and not reveal
        print(f"  {i}/{len(questions)}  {q['id']}  " + ("(held-out)" if hidden else out["status"]), flush=True)
        if out["status"] == "quota_exhausted":
            print("\nStopping early: " + out["answer"])
            stopped = True
            continue
        if out["input_tokens"]:
            time.sleep(PAUSE)

    not_attempted = len(questions) - len(rows)
    dev = [r for r in rows if r["split"] != "heldout"]
    held = [r for r in rows if r["split"] == "heldout"]
    dev_summary = summarize(dev)
    held_summary = summarize(held) if held else None

    print_summary(f"Run: {run_name}  (dev questions, model {llm.MODEL})", dev_summary, not_attempted)
    print_problems(dev)
    if held_summary:
        print_summary("HELD-OUT questions (totals only, never tune on these)", held_summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "run": run_name, "date": str(date.today()),
        "config": {"llm": llm.MODEL, "retrieval": RETRIEVAL_MODEL, "n_questions": len(questions)},
        "not_attempted": not_attempted,
        "summary": dev_summary, "heldout_summary": held_summary,
        "details": dev, "heldout_details": held if reveal else "hidden",
    }
    out_path = RESULTS_DIR / f"{run_name}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved:", out_path)