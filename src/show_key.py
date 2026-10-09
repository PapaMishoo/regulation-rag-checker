import sys

from embed_chunks import load_chunks
from questions import load_questions

# Optional: start from a given question, like  python src/show_key.py q34
start = sys.argv[1] if len(sys.argv) > 1 else None
started = start is None

first_chunk = {}
for chunk in load_chunks():
    first_chunk.setdefault(chunk["label"], chunk["text"])

not_found = 0
for q in load_questions():
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