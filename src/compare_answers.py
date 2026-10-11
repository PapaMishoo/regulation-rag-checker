import json
from pathlib import Path


def frac(items, key):
    return f"{sum(1 for r in items if r.get(key))}/{len(items)}"


for path in sorted(Path("eval/answer_results").glob("*.json")):
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in d["details"] if not r.get("failed")]
    real = [r for r in rows if r["expected"]]
    if not real:
        continue
    off = [r for r in rows if not r["expected"]]
    present = [r for r in real if r.get("in_passages")]
    missing = [r for r in real if not r.get("in_passages")]
    print("=" * 60)
    print(d["run"], "| model:", d["config"]["llm"], "| dev questions:", len(rows))
    print("  Cited the expected article:                          ", frac(real, "cited_expected"))
    print("  Refused when the right article WAS in the passages:  ", frac(present, "false_refusal"))
    print("  Refused when the right article was NOT in them:      ", frac(missing, "false_refusal"))
    print("  Off-topic questions refused:                         ", frac(off, "correctly_refused"))
    print("  Answers with an invalid citation:                    ", sum(1 for r in rows if r["bad_citations"]))
    h = d.get("heldout_summary")
    if h:
        print(f"  Held-out: cited expected {h['cited_expected_of_all_real']:.0%}, "
              f"off-topic refused {h['off_topic_refused']:.0%}, wrongly refused {h['false_refusal_rate']:.0%}")