import sys

from embed_chunks import load_chunks
from questions import load_fresh, load_questions

args = [a for a in sys.argv[1:] if not a.startswith("--")]
fresh = "--fresh" in sys.argv
start = args[0] if args else None
started = start is None

first_chunk = {}
for chunk in load_chunks():
    first_chunk.setdefault(chunk["label"], chunk["text"])

not_found = 0
for q in (load_fresh() if fresh else load_questions()):
    if q["id"] == start:
        started = True
    if not started:
        continue
    print(f"\n[{q['id']}] ({q['type']}, {q['split']}) {q['question']}")
    if not q["expected"]:
        print("   (out of scope: nothing should match)")
    for label in q["expected"]:
        text = first_chunk.get(label)
        if text is None:
            not_found += 1
        print(f"   -> {label}: " + (text[:150] + "..." if text else "LABEL NOT FOUND"))

print("\nLabels not found:", not_found)