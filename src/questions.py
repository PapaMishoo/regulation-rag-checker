import json
from pathlib import Path

QUESTION_FILES = [Path("eval/questions.json"), Path("eval/questions_extra.json")]


def load_questions():
    questions = []
    for path in QUESTION_FILES:
        if path.exists():
            questions.extend(json.loads(path.read_text(encoding="utf-8")))
    for q in questions:
        q.setdefault("split", "dev")  # the original questions have no split, so they count as dev
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate question ids found in the question files.")
    return questions