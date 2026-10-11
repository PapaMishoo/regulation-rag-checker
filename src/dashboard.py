import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RESULTS_DIR = Path("eval/results")

# Friendly names, in story order. Any other run still shows up under its own name.
NAMES = {
    "v2_minilm_base": "1. Baseline: small model, recitals included",
    "v2_minilm_norecitals": "2. Recitals left out of search",
    "v2_bge_base": "3. Stronger model (bge-base)",
    "v2_bge_rerank": "4. Plus a re-ranker (rejected)",
    "v2_bge_markers": "5. Amendment markers removed",
    "v2_bge_markers_merge": "6. Short chunks merged (adopted)",
    "v2_bge_chunk800": "7. 800-character chunks",
    "v2_bge_chunk2000": "8. 2,000-character chunks",
}
HIDDEN = {"pc_check"}  # reproduction checks, not experiments
BASELINE = "v2_minilm_base"
ADOPTED = "v2_bge_markers_merge"


@st.cache_data
def load_runs():
    runs = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("run") in HIDDEN or "summary" not in data:
            continue
        runs[data["run"]] = data
    return runs


def label(run):
    return NAMES.get(run, run)


st.set_page_config(page_title="Regulation RAG Checker: evaluation", layout="wide")
runs = load_runs()
if not runs:
    st.error("No result files found in eval/results. Run python src/evaluate.py first.")
    st.stop()
order = [r for r in NAMES if r in runs] + [r for r in runs if r not in NAMES]

st.title("Regulation RAG Checker: evaluation dashboard")
st.write(
    "A search system over the GDPR and the EU AI Act. This dashboard shows how well the "
    "retrieval stage finds the right article, and how each experiment changed that. "
    "It measures retrieval only, and it is not legal advice."
)

# Headline numbers for the adopted setup
if ADOPTED in runs and BASELINE in runs:
    a, b = runs[ADOPTED]["summary"], runs[BASELINE]["summary"]
    st.subheader("Where it ended up, compared with the starting point")
    cols = st.columns(4)
    for col, (name, key) in zip(cols, [("Hit@1", "hit_at_1"), ("Hit@3", "hit_at_3"), ("Hit@5", "hit_at_5")]):
        col.metric(name, f"{a[key]:.0%}", f"{(a[key] - b[key]) * 100:+.0f} pts vs baseline")
    cols[3].metric("MRR", f"{a['mrr']:.2f}", f"{a['mrr'] - b['mrr']:+.2f} vs baseline")
    st.caption(
        f"{a['n_scored']} development questions with a known answer. Hit@3 means the right article "
        "was among the top 3 results. MRR rewards ranking it higher (1.0 means always first)."
    )

# Experiment history
st.subheader("Every experiment")
rows = []
for r in order:
    s, h = runs[r]["summary"], runs[r].get("heldout_summary")
    rows.append({
        "Experiment": label(r),
        "Hit@1": s["hit_at_1"], "Hit@3": s["hit_at_3"], "Hit@5": s["hit_at_5"], "MRR": s["mrr"],
        "Held-out Hit@3": h["hit_at_3"] if h else None,
        "Chunks searched": runs[r].get("config", {}).get("chunks"),
    })
df = pd.DataFrame(rows)
st.dataframe(
    df.style.format({"Hit@1": "{:.0%}", "Hit@3": "{:.0%}", "Hit@5": "{:.0%}", "MRR": "{:.2f}",
                     "Held-out Hit@3": "{:.0%}"}, na_rep="-"),
    hide_index=True, use_container_width=True,
)

metric = st.selectbox("Chart this measure", ["Hit@3", "Hit@1", "Hit@5", "MRR"])
fmt = ".2f" if metric == "MRR" else ".0%"
chart_df = df[["Experiment", metric]].rename(columns={metric: "Value"})
main_chart = (
    alt.Chart(chart_df)
    .mark_bar()
    .encode(
        y=alt.Y("Experiment:N", sort=None, title=None),
        x=alt.X("Value:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=fmt), title=metric),
        tooltip=["Experiment", alt.Tooltip("Value:Q", format=fmt, title=metric)],
    )
    .properties(height=300)
)
st.altair_chart(main_chart, use_container_width=True)
st.caption(
    "The held-out group is only 8 questions with a known answer, so it is a sanity check, not a precise "
    "measurement. Differences of one question (about 2 points on the development set) are noise."
)

# By question type
st.subheader("Results by question type (Hit@3)")
picks = st.multiselect("Compare experiments", order, default=[r for r in (BASELINE, ADOPTED) if r in runs],
                       format_func=label)
if picks:
    type_rows = []
    for r in picks:
        for t, v in runs[r]["summary"]["by_type"].items():
            type_rows.append({"Experiment": label(r), "Question type": t, "Hit@3": v["hit_at_3"]})
    type_df = pd.DataFrame(type_rows)
    type_chart = (
        alt.Chart(type_df)
        .mark_bar()
        .encode(
            x=alt.X("Experiment:N", axis=None, sort=None),
            y=alt.Y("Hit@3:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
            color=alt.Color("Experiment:N", sort=None, legend=alt.Legend(orient="bottom", title=None)),
            column=alt.Column("Question type:N", title=None),
            tooltip=["Experiment", "Question type", alt.Tooltip("Hit@3:Q", format=".0%")],
        )
        .properties(width=160, height=220)
    )
    st.altair_chart(type_chart)
    st.caption("direct = worded like the law, plain = everyday wording, multi = answer spread over several articles.")

# Refusal cutoff
st.subheader("The 'I don't know' cutoff")
cut_run = st.selectbox("Experiment", order, index=order.index(ADOPTED) if ADOPTED in order else 0,
                       format_func=label, key="cutoff")
cut_rows = []
for t, v in runs[cut_run]["summary"]["thresholds"].items():
    cut_rows.append({"Cutoff": float(t), "Measure": "Real questions kept", "Share": v["real_questions_kept"]})
    cut_rows.append({"Cutoff": float(t), "Measure": "Off-topic questions refused", "Share": v["out_of_scope_refused"]})
cut_df = pd.DataFrame(cut_rows)
lines = (
    alt.Chart(cut_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("Cutoff:Q", scale=alt.Scale(domain=[0.2, 0.7]), axis=alt.Axis(format=".2f")),
        y=alt.Y("Share:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%"), title=None),
        color=alt.Color("Measure:N", scale=alt.Scale(range=["#2ca02c", "#d62728"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Measure", "Cutoff", alt.Tooltip("Share:Q", format=".0%")],
    )
)
chosen = alt.Chart(pd.DataFrame({"Cutoff": [0.55]})).mark_rule(strokeDash=[4, 4]).encode(x="Cutoff:Q")
st.altair_chart((lines + chosen).properties(height=300), use_container_width=True)
st.caption(
    "If the best search score is below the cutoff, the system refuses. The dashed line is the 0.55 cutoff I "
    "use. No cutoff separates the two groups cleanly: questions about other countries' privacy laws score "
    "high, so the answer step has to catch those."
)

# Misses explorer
st.subheader("Questions it still gets wrong")
miss_run = st.selectbox("Experiment", order, index=order.index(ADOPTED) if ADOPTED in order else 0,
                        format_func=label, key="misses")
miss_rows = []
for d in runs[miss_run]["details"]:
    if d["expected"] and (d["rank"] is None or d["rank"] > 3):
        miss_rows.append({
            "ID": d["id"], "Type": d["type"], "Question": d["question"],
            "Right article(s)": ", ".join(d["expected"]),
            "Got first": d["retrieved"][0]["label"],
            "Rank of the right article in the search": d["pool_rank"] if d["pool_rank"] else "not in top 20",
        })
st.dataframe(pd.DataFrame(miss_rows), hide_index=True, use_container_width=True)
with st.expander("Settings used for this run"):
    st.json(runs[miss_run].get("config", {}))

st.subheader("Honest limits")
st.markdown(
    "- The test set is small (52 questions with a known answer), so small differences are noise.\n"
    "- The answer key was written by hand from my own reading of the laws.\n"
    "- Most remaining misses are cases where everyday wording differs from legal wording.\n"
    "- Source texts are from EUR-Lex (consolidated AI Act of 27 July 2026), reused with credit."
)