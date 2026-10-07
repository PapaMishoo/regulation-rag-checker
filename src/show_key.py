import json
from pathlib import Path

from embed_chunks import load_chunks

questions = json.loads(Path("eval/questions.json").read_text(encoding="utf-8"))

# The first chunk of every article, so we can peek at its opening
first_chunk = {}
for chunk in load_chunks():
    first_chunk.setdefault(chunk["label"], chunk["text"])

for q in questions:
    print(f"\n[{q['id']}] ({q['type']}) {q['question']}")
    if not q["expected"]:
        print("   (out of scope: nothing should match)")
    for label in q["expected"]:
        text = first_chunk.get(label)
        print(f"   -> {label}: " + (text[:150] + "..." if text else "LABEL NOT FOUND"))