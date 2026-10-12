import json
import time
from datetime import date
from pathlib import Path

import llm
from answer import RETRIEVAL_MODEL, answer_question
from embed_chunks import load_index
from embedders import Embedder
from evaluate import first_hit_rank
from evaluate_answers import PAUSE, judge, print_problems, print_summary
from evaluate_answers import summarize as summarize_answers
from questions import load_fresh
from rewrite import get_rewrite, search_with_rewrite
from search import search

POOL = 20


def retrieval_line(name, rows):
    real = [r for r in rows if r["expected"]]
    hit = lambda k: sum(1 for r in real if r["rank"] and r["rank"] <= k)
    mrr = sum(1 / r["rank"] for r in real if r["rank"]) / len(real)
    print(f"  {name:<28} Hit@1 {hit(1)}/{len(real)}   Hit@3 {hit(3)}/{len(real)}   Hit@5 {hit(5)}/{len(real)}   MRR {mrr:.2f}")


if __name__ == "__main__":
    questions = load_fresh()
    chunks, embeddings = load_index(RETRIEVAL_MODEL, include_recitals=False)
    embedder = Embedder(RETRIEVAL_MODEL)

    plain_rows, rewrite_rows = [], []
    for q in questions:
        expected = q["expected"]
        base = search(q["question"], embedder, chunks, embeddings, top_k=POOL)
        rewritten = get_rewrite(q["question"])
        boosted, _ = search_with_rewrite(q["question"], rewritten, embedder, chunks, embeddings, POOL)
        for rows, results, extra in ((plain_rows, base, {}), (rewrite_rows, boosted, {"rewrite": rewritten})):
            rows.append({"id": q["id"], "question": q["question"], "expected": expected,
                         "rank": first_hit_rank(results[:5], expected) if expected else None,
                         "pool_rank": first_hit_rank(results, expected) if expected else None,
                         "got": results[0][0]["label"], **extra})

    print("=" * 60)
    print("RETRIEVAL on the fresh questions (right article in the top 1 / 3 / 5)")
    retrieval_line("before rewriting", plain_rows)
    retrieval_line("with rewriting (final)", rewrite_rows)
    print("\nMisses with rewriting (right article not in the top 3):")
    for r in rewrite_rows:
        if r["expected"] and (r["rank"] is None or r["rank"] > 3):
            print(f"  [{r['id']}] {r['question']}")
            print(f"       wanted: {', '.join(r['expected'])}   got #1: {r['got']}   search rank: {r['pool_rank'] or 'not in top 20'}")
            print(f"       rewrite: {r['rewrite']}")

    print("=" * 60)
    print("ANSWERS on the fresh questions (final setup)")
    answer_rows = []
    for i, q in enumerate(questions, start=1):
        out = answer_question(q["question"], embedder, chunks, embeddings)
        answer_rows.append(judge(q, out))
        print(f"  {i}/{len(questions)}  {q['id']}  {out['status']}", flush=True)
        if out["input_tokens"]:
            time.sleep(PAUSE)
    summary = summarize_answers(answer_rows)
    print_summary("Fresh questions, final setup", summary)
    print_problems(answer_rows)

    out_path = Path("eval") / "fresh_results.json"
    out_path.write_text(json.dumps({
        "date": str(date.today()), "llm": llm.MODEL,
        "retrieval_before_rewriting": plain_rows, "retrieval_with_rewriting": rewrite_rows,
        "answers": answer_rows, "answer_summary": summary,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved:", out_path)