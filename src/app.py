import html
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv

from build_report import REDTEAM_REVIEW
from embed_chunks import chunk_to_text, load_index
from embedders import Embedder
from search import search

load_dotenv()

RETRIEVAL_MODEL = "bge-base"
CUTOFF = 0.55
PROCESSED = Path("data/processed")
REWRITES_PATH = Path("eval/rewrites_gemini-3.5-flash-lite_plain.json")
ANSWERS_PATH = Path("eval/answer_results/answers_final.json")
FRESH_PATH = Path("eval/fresh_results.json")
REDTEAM_PATHS = [Path("eval/redteam_results_round2.json"), Path("eval/redteam_results.json")]
EXAMPLES = [
    "How long can a company keep my personal data?",
    "Can I move my data from one service to another?",
    "Which AI practices are banned?",
    "What does HIPAA require for medical records?",
]
STATUS = {
    "answered": ("Answered with citations", "ok"),
    "answered_without_valid_citation": ("Answered, but no valid citation", "stop"),
    "refused_low_score": ("Refused at gate 1: low score", "stop"),
    "refused_by_model": ("Refused: the model says it is not covered", "stop"),
    "api_error": ("The model call failed", "stop"),
    "quota_exhausted": ("Free-tier quota used up", "stop"),
}

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, .stApp, [class*="css"] {font-family: "IBM Plex Sans", system-ui, sans-serif;}
.stApp {background: #15130f; color: #f0e9d8;}
#MainMenu, footer, header {visibility: hidden;}
.block-container {max-width: 1100px; padding-top: 2.2rem; padding-bottom: 4rem; border-top: 6px double #f0e9d8; margin-top: 1.2rem;}
.eyebrow {display:inline-block; font:500 12px "IBM Plex Mono",monospace; letter-spacing:.14em; text-transform:uppercase; color:#e8643f; padding:0 0 9px; border-bottom:1px solid #e8643f;}
.hero-title {font:800 clamp(38px,6vw,68px)/1 Fraunces,Georgia,serif; letter-spacing:-.03em; margin:20px 0 12px; color:#f0e9d8;}
.grad {font-style:italic; font-weight:600; color:#e8643f;}
.lede {color:#cfc7b3; font-size:17.5px; max-width:840px; margin-bottom:18px;}
.sec {color:#cfc7b3; font-size:15px; margin:4px 0 14px;}
.trace {display:grid; grid-template-columns:repeat(6,1fr); border:1px solid #3a3629; background:#1e1b15; margin:14px 0 18px;}
.tstep {padding:12px; border-right:1px solid #3a3629; font-size:12.5px; color:#a39c88; line-height:1.4;}
.tstep:last-child {border-right:0;}
.tstep b {display:block; color:#f0e9d8; font:600 14.5px Fraunces,Georgia,serif; margin:8px 0 3px;}
.tn {width:22px; height:22px; border:1.5px solid #f0e9d8; border-radius:50%; display:grid; place-items:center; font:600 11px "IBM Plex Mono",monospace; color:#f0e9d8;}
.tstep.ok .tn {background:#7bc99a; border-color:#7bc99a; color:#15130f;}
.tstep.stop {background:rgba(232,100,63,.12);}
.tstep.stop .tn {background:#e8643f; border-color:#e8643f; color:#15130f;}
.tstep.skip {opacity:.55;}
.card {padding:16px 20px; border:1px solid #3a3629; border-left:4px solid #8db4dd; background:#1e1b15; margin-bottom:12px;}
.card .rk {font:600 12px "IBM Plex Mono",monospace; color:#e8643f;}
.card h4 {margin:4px 0 4px; font:600 18px Fraunces,Georgia,serif; color:#f0e9d8;}
.bar {height:6px; background:#2a261e; margin:8px 0;}
.bar i {display:block; height:100%; background:#8db4dd;}
.snip {color:#cfc7b3; font-size:14px;}
.pill {display:inline-block; padding:3px 9px; border-radius:2px; font:500 12px "IBM Plex Mono",monospace; border:1px solid; margin:2px 6px 2px 0; letter-spacing:.02em;}
.pill.ok {color:#7bc99a; border-color:#7bc99a; background:rgba(123,201,154,.12);}
.pill.stop {color:#e3b04b; border-color:#e3b04b; background:rgba(227,176,75,.12);}
.pill.muted {color:#a39c88; border-color:#6d6757;}
.q {font:600 24px/1.25 Fraunces,Georgia,serif; margin:8px 0 2px; letter-spacing:-.01em; color:#f0e9d8;}
.answer {padding:18px 22px; border-left:4px solid #e8643f; background:#1e1b15; font:500 17px/1.55 Fraunces,Georgia,serif; color:#f0e9d8; margin:10px 0 12px;}
.small {color:#a39c88; font-size:13px;}
@media (max-width:900px) {.trace {grid-template-columns:repeat(2,1fr);}}
</style>"""

HERO = "".join([
    '<span class="eyebrow">Portfolio project &middot; Retrieval-augmented generation</span>',
    '<div class="hero-title">Regulation <span class="grad">RAG Checker</span></div>',
    '<div class="lede">Ask a question about the <b>GDPR</b> or the <b>EU AI Act</b>. The system finds the article that answers it, '
    'writes a short cited answer, and refuses when the texts do not cover the question. Try the live search, or read real recorded outputs. '
    'Informational only, not legal advice.</div>',
])


def esc(text):
    return html.escape(str(text if text is not None else ""))


def pill(text, tone="muted"):
    return f'<span class="pill {tone}">{esc(text)}</span>'


def trace(steps):
    cells = "".join(
        f'<div class="tstep {state}"><div class="tn">{i}</div><b>{esc(title)}</b>{esc(detail)}</div>'
        for i, (title, detail, state) in enumerate(steps, start=1))
    st.markdown(f'<div class="trace">{cells}</div>', unsafe_allow_html=True)


def ensure_index():
    """On a fresh clone the search index does not exist yet, so build it once."""
    chunks_path = PROCESSED / "chunks.jsonl"
    embeddings_path = PROCESSED / f"embeddings_{RETRIEVAL_MODEL}.npy"
    if chunks_path.exists() and embeddings_path.exists():
        return
    from build_chunks import build_chunks
    chunks = build_chunks(max_chars=1500, strip_markers=True, merge_short=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (PROCESSED / "chunks_config.json").write_text(
        json.dumps({"max_chars": 1500, "strip_markers": True, "merge_short": True}, indent=2), encoding="utf-8")
    embeddings = Embedder(RETRIEVAL_MODEL).encode_docs([chunk_to_text(c) for c in chunks])
    np.save(embeddings_path, embeddings)


@st.cache_resource(show_spinner=False)
def load_system():
    ensure_index()
    chunks, embeddings = load_index(RETRIEVAL_MODEL, include_recitals=False)
    embedder = Embedder(RETRIEVAL_MODEL)
    texts = {}
    for c in chunks:
        texts.setdefault(c["label"], []).append(c["text"])
    articles = {label: " ".join(parts) for label, parts in texts.items()}
    return chunks, embeddings, embedder, articles


def top_articles(results, k=5):
    seen, out = set(), []
    for chunk, score in results:
        if chunk["label"] in seen:
            continue
        seen.add(chunk["label"])
        out.append((chunk, score))
        if len(out) == k:
            break
    return out


@st.cache_data
def load_recorded():
    rewrites = json.loads(REWRITES_PATH.read_text(encoding="utf-8")) if REWRITES_PATH.exists() else {}

    def item(r, rewrite=None, review=None):
        return {"id": r["id"], "question": r.get("question") or r.get("prompt"), "status": r["status"],
                "answer": r["answer"], "citations": r.get("citations", []), "expected": r.get("expected"),
                "sources": r.get("sources", []), "top_score": r.get("top_score"),
                "rewrite": rewrite or r.get("rewrite"), "review": review}

    groups = {}
    if ANSWERS_PATH.exists():
        rows = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))["details"]
        groups["Development questions (55)"] = [item(r, rewrites.get(r["question"])) for r in rows]
    if FRESH_PATH.exists():
        rows = json.loads(FRESH_PATH.read_text(encoding="utf-8"))["answers"]
        groups["Fresh test (17)"] = [item(r, rewrites.get(r["question"])) for r in rows]
    for path in REDTEAM_PATHS:
        if path.exists():
            rows = json.loads(path.read_text(encoding="utf-8"))["results"]
            groups["Attacks on the system (20)"] = [
                item(r, review=REDTEAM_REVIEW.get(r["id"], ("", ""))[1]) for r in rows]
            break
    return groups


def show_item(item, articles):
    status = item["status"]
    label, tone = STATUS.get(status, (status, "muted"))
    gate1_ok = status != "refused_low_score"
    ran_answer = status in ("answered", "answered_without_valid_citation", "refused_by_model", "api_error")
    gate2 = {"answered": ("citations checked", "ok"), "refused_by_model": ("model said: not covered", "stop"),
             "answered_without_valid_citation": ("no valid citation", "stop")}.get(status, ("not reached", "skip"))
    rewrite = item.get("rewrite")
    top = item.get("top_score")
    st.markdown(f'<div class="q">{esc(item["question"])}</div>', unsafe_allow_html=True)
    trace([
        ("Question", "received", "ok"),
        ("Rewrite", (rewrite[:80] + ("..." if len(rewrite) > 80 else "")) if rewrite else "not recorded", "ok" if rewrite else "skip"),
        ("Search", f"best score {top:.2f}" if top is not None else "done", "ok"),
        ("Gate 1: cutoff", ("passed" if gate1_ok else "blocked") + f" (cutoff {CUTOFF})", "ok" if gate1_ok else "stop"),
        ("Answer", "written from the top 5 passages" if ran_answer else "skipped", "ok" if ran_answer else "skip"),
        ("Gate 2: checks", gate2[0], gate2[1]),
    ])
    pills = pill(label, tone)
    expected = item.get("expected")
    if expected is not None:
        if expected:
            hit = any(c in expected for c in item["citations"])
            pills += pill("Cited the expected article" if hit else "Did not cite the expected article", "ok" if hit else "stop")
        else:
            pills += pill("Out of scope: nothing should match", "muted")
    st.markdown(pills, unsafe_allow_html=True)
    st.markdown(f'<div class="answer">{esc(item["answer"])}</div>', unsafe_allow_html=True)
    if item["citations"]:
        st.markdown("**Cited:** " + " ".join(pill(c, "ok") for c in item["citations"]), unsafe_allow_html=True)
    if expected:
        st.markdown('<span class="small">Expected article(s): ' + esc(", ".join(expected)) + "</span>", unsafe_allow_html=True)
    if item.get("review"):
        st.markdown('<span class="small">My review: ' + esc(item["review"]) + "</span>", unsafe_allow_html=True)
    if item["sources"]:
        st.markdown('<span class="small">Passages the model was given: ' + esc(", ".join(dict.fromkeys(item["sources"]))) + "</span>",
                    unsafe_allow_html=True)
    for cited in item["citations"]:
        with st.expander(f"Check it yourself: read {cited}"):
            st.write(articles.get(cited, "(text not found)"))
    st.caption("AI-generated from the GDPR and EU AI Act texts. General information, not legal advice. "
               "Check the cited articles and consult a qualified professional.")


def live_tab(chunks, embeddings, embedder, articles):
    st.markdown('<div class="sec">Type any question. The search below runs for real, right now, on your question. '
                'Question rewriting and answer writing need a language model, so they appear in the recorded runs.</div>',
                unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        if col.button(example, use_container_width=True):
            st.session_state["q"] = example
    question = st.text_input("Your question", key="q", placeholder="For example: Can I ask a company to stop using my data?")
    if not question.strip():
        return

    if os.getenv("GEMINI_API_KEY") and st.toggle("Run the full system (rewrite + answer) with the Gemini key in .env"):
        from answer import answer_question
        with st.spinner("Running the full pipeline..."):
            out = answer_question(question, embedder, chunks, embeddings, save_rewrite=False)
        show_item({"question": question, "status": out["status"], "answer": out["answer"], "citations": out["citations"],
                   "expected": None, "sources": out["sources"], "top_score": out["top_score"], "rewrite": out.get("rewrite")}, articles)
        return

    results = top_articles(search(question, embedder, chunks, embeddings, top_k=20), 5)
    best = results[0][1]
    passes = best >= CUTOFF
    trace([
        ("Question", "typed by you", "ok"),
        ("Rewrite", "needs a language model (off here)", "skip"),
        ("Search", f"best score {best:.2f}", "ok"),
        ("Gate 1: cutoff", ("passed" if passes else "blocked") + f" (cutoff {CUTOFF})", "ok" if passes else "stop"),
        ("Answer", "needs a language model (off here)", "skip"),
        ("Gate 2: checks", "not run", "skip"),
    ])
    if passes:
        st.markdown(pill("Passes gate 1: the full system would now write an answer from these passages", "ok"), unsafe_allow_html=True)
    else:
        st.markdown(pill("Blocked at gate 1: the full system would answer that the documents do not cover this", "stop"),
                    unsafe_allow_html=True)
    for rank, (chunk, score) in enumerate(results, start=1):
        st.markdown(
            f'<div class="card"><span class="rk">#{rank}</span><h4>{esc(chunk["label"])}</h4>'
            f'<div class="bar"><i style="width:{max(0.0, min(1.0, score)) * 100:.0f}%"></i></div>'
            f'<span class="small">similarity {score:.2f}</span><div class="snip">{esc(chunk["text"][:280])}...</div></div>',
            unsafe_allow_html=True)
        with st.expander(f"Read the whole of {chunk['label']}"):
            st.write(articles.get(chunk["label"], ""))
    st.markdown('<div class="small">Without question rewriting, everyday wording can miss the right article. Try "Can I move my data from '
                'one service to another?", then find that question in the recorded runs to see the rewritten version.</div>',
                unsafe_allow_html=True)


def recorded_tab(articles):
    groups = load_recorded()
    if not groups:
        st.info("No recorded results found in the eval folder.")
        return
    st.markdown('<div class="sec">Real outputs of the full system, saved in the repository. Pick any one, follow it through the six stages, '
                'and open the cited article to check the answer yourself.</div>', unsafe_allow_html=True)
    group = st.radio("Which set?", list(groups), horizontal=True)
    items = groups[group]
    labels = {i["id"]: f'{i["id"]}: {i["question"][:95]}' for i in items}
    pick = st.selectbox("Question", list(labels), format_func=lambda k: labels[k])
    show_item(next(i for i in items if i["id"] == pick), articles)


def about_tab():
    st.markdown("""
**What this is.** A retrieval-augmented generation system built from pretrained models: an embedding model finds the right article, a hosted language model writes a short answer from the retrieved passages, and my own code decides when to refuse and checks every citation. I did not train a model.

**What I measured.** 44 development questions, 13 fresh questions I had not tuned on, 11 held-out questions, and 20 attacks on the system. The full evidence, including the experiments that did not work, is in the report page in the repository.

**What it gets wrong.** It is sometimes over-cautious, answers can be incomplete when a long article is split into chunks, and it handles English only.

**Run it yourself.** Clone the repository, install the requirements, and run this app. With a free Gemini key in a `.env` file, the toggle in the first tab switches on the full pipeline.
""")


st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HERO, unsafe_allow_html=True)
with st.spinner("Loading the search index (the first launch on a new machine builds it and can take a few minutes)..."):
    chunks, embeddings, embedder, articles = load_system()
tab1, tab2, tab3 = st.tabs(["Try a question", "Recorded runs", "About"])
with tab1:
    live_tab(chunks, embeddings, embedder, articles)
with tab2:
    recorded_tab(articles)
with tab3:
    about_tab()