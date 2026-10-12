import json
import sys
from pathlib import Path

from embed_chunks import load_chunks
from questions import load_questions

runs = sys.argv[1:]
facts = json.loads(Path("eval/keyfacts.json").read_text(encoding="utf-8"))
questions = {q["id"]: q for q in load_questions()}

article_text = {}
for chunk in load_chunks():
    article_text.setdefault(chunk["label"], []).append(chunk["text"])
article_text = {label: " ".join(parts).lower() for label, parts in article_text.items()}


def found(text, alternatives):
    return any(a.lower() in text for a in alternatives)


print("Checking that each key fact really appears in its article:")
bad = 0
for entry in facts:
    label = questions[entry["id"]]["expected"][0]
    for alternatives in entry["facts"]:
        if not found(article_text.get(label, ""), alternatives):
            bad += 1
            print(f"  NOT FOUND in {label}: {alternatives}  (question {entry['id']})")
print("  all key facts are in their articles" if not bad else f"  {bad} key fact(s) need fixing")

for name in runs:
    data = json.loads((Path("eval/answer_results") / f"{name}.json").read_text(encoding="utf-8"))
    answers = {r["id"]: r for r in data["details"]}
    print("=" * 60)
    print(name)
    total_found = total = 0
    for entry in facts:
        r = answers.get(entry["id"])
        text = r["answer"].lower() if r and not r.get("failed") else ""
        hits = [found(text, alternatives) for alternatives in entry["facts"]]
        total_found += sum(hits)
        total += len(hits)
        print(f"  {entry['id']}: {sum(hits)}/{len(hits)} key facts   (status: {r['status'] if r else 'missing'})")
    print(f"  TOTAL: {total_found}/{total}")