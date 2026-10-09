import json
from pathlib import Path

print(f"{'run':<26}{'n':>4}{'Hit@1':>7}{'Hit@3':>7}{'Hit@5':>7}{'MRR':>6}   held-out: n / Hit@3 / MRR")
for path in sorted(Path("eval/results").glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    s = data["summary"]
    h = data.get("heldout_summary")
    held = f"{h['n_scored']} / {h['hit_at_3']:.0%} / {h['mrr']:.2f}" if h else "-"
    tag = " (re-ranked)" if data.get("config", {}).get("reranker") else ""
    print(f"{data['run'] + tag:<26}{s['n_scored']:>4}{s['hit_at_1']:>7.0%}{s['hit_at_3']:>7.0%}"
          f"{s['hit_at_5']:>7.0%}{s['mrr']:>6.2f}   {held}")