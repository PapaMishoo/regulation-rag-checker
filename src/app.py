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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
html, body, .stApp, [class*="css"] {font-family: Inter, system-ui, sans-serif;}
.stApp {background: radial-gradient(900px 520px at 10% -6%, rgba(129,140,248,.22), transparent 60%), radial-gradient(800px 480px at 96% 6%, rgba(94,234,212,.15), transparent 60%), #060914;}
#MainMenu, footer, header {visibility: hidden;}
.block-container {max-width: 1100px; padding-top: 2.2rem; padding-bottom: 4rem;}
.eyebrow {display:inline-block; font-size:12px; font-weight:600; letter-spacing:.14em; text-transform:uppercase; color:#5eead4; padding:8px 14px; border:1px solid rgba(94,234,212,.35); border-radius:999px; background:rgba(94,234,212,.07);}
.hero-title {font-size:clamp(34px,5vw,60px); font-weight:800; letter-spacing:-.03em; line-height:1.05; margin:16px 0 10px;}
.grad {background:linear-gradient(95deg,#5eead4,#818cf8 55%,#f472b6); -webkit-background-clip:text; background-clip:text; color:transparent;}
.lede {color:#b9c3d8; font-size:17px; max-width:840px; margin-bottom:18px;}
.sec {color:#9aa6bd; font-size:14.5px; margin:4px 0 14px;}
.trace {display:grid; grid-template-columns:repeat(6,1fr); gap:10px; margin:14px 0 18px;}
.tstep {padding:12px; border-radius:14px; border:1px solid rgba(255,255,255,.09); background:rgba(255,255,255,.04); font-size:12px; color:#9aa6bd; line-height:1.35;}
.tstep b {display:block; color:#e8ecf5; font-size:13px; margin:7px 0 3px;}
.tn {width:22px; height:22px; border-radius:50%; display:grid; place-items:center; font:700 11px "JetBrains Mono",monospace; background:#334155; color:#fff;}
.tstep.ok {border-color:rgba(94,234,212,.4);}
.tstep.ok .tn {background:linear-gradient(135deg,#5eead4,#818cf8); color:#06101c;}
.tstep.stop {border-color:rgba(251,191,36,.5); background:rgba(251,191,36,.06);}
.tstep.stop .tn {background:#fbbf24; color:#06101c;}
.tstep.skip {opacity:.5;}
.card {padding:16px 20px; border-radius:16px; border:1px solid rgba(255,255,255,.09); background:rgba(255,255,255,.045); margin-bottom:12px;}
.card .rk {font:700 12px "JetBrains Mono",monospace; color:#818cf8;}
.card h4 {margin:4px 0 4px; font-size:16px; font-weight:700; color:#e8ecf5;}
.bar {height:6px; border-radius:99px; background:rgba(255,255,255,.07); overflow:hidden; margin:8px 0;}
.bar i {display:block; height:100%; background:linear-gradient(90deg,#818cf8,#5eead4);}
.snip {color:#aab4c9; font-size:13.5px;}
.pill {display:inline-block; padding:4px 11px; border-radius:99px; font-size:12px; font-weight:600; border:1px solid; margin:2px 6px 2px 0;}
.pill.ok {color:#34d399; border-color:rgba(52,211,153,.45); background:rgba(52,211,153,.09);}
.pill.stop {color:#fbbf24; border-color:rgba(251,191,36,.5); background:rgba(251,191,36,.08);}
.pill.muted {color:#cbd5e1; border-color:rgba(203,213,225,.3);}
.q {font-size:21px; font-weight:700; margin:6px 0 2px; letter-spacing:-.01em;}
.answer {padding:18px 20px; border-radius:16px; border-left:4px solid #818cf8; background:rgba(129,140,248,.09); font-size:15.5px; color:#e8ecf5; margin:10px 0 12px;}
.small {color:#8b96ad; font-size:12.5px;}
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