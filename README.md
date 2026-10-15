# Regulation RAG Checker

Ask a question about the **GDPR** or the **EU AI Act** in plain English. The system finds the article that answers it, writes a short answer that cites its sources, and refuses when the texts don't cover the question.

This repository is as much about **measuring** the system as building it: a hand-written test set, ten measured retrieval experiments (including the ones that failed), a fresh test I hadn't tuned on, and twenty attacks on my own system.

**Live report:** https://papamishoo.github.io/regulation-rag-checker/report/

> **What it is, and isn't.** A retrieval-augmented generation (RAG) system built from pretrained models. An embedding model finds the right article, a hosted language model writes the answer from the retrieved text, and my own code decides when to refuse and checks every citation. I did not train a model. It is informational only, **not legal advice**, and not an official EU service.

![Report overview](docs/images/report-hero.png)

## Results at a glance

| | First version | Final setup |
|---|---|---|
| Right article ranked **first** (44 development questions) | 41% | **89%** |
| Right article in the **top 3** | 61% | **93%** |
| Answers that cite the right article | 37 / 44 | **41 / 44** |
| Off-topic questions refused (including near-misses like HIPAA and CCPA) | 11 / 11 | **11 / 11** |
| Attacks on the system that got through | | **0 / 20** |

On a **fresh test** of 13 new questions I hadn't tuned on, the right article came first for 12 of 13 (8 of 13 before question rewriting). Rewriting did not improve the top-3 count there, which is reported honestly below.

![The ten experiments](docs/images/report-journey.png)

## How it works

```mermaid
flowchart LR
    Q[Question] --> R[Rewrite in legal vocabulary]
    R --> S[Search articles with embeddings]
    S --> G1{Gate 1: best score at least 0.55?}
    G1 -- no --> X[Refuse]
    G1 -- yes --> A[Answer from the top 5 passages, citing source numbers]
    A --> G2{Gate 2: covered? citations valid?}
    G2 -- no --> X
    G2 -- yes --> O[Cited answer]
```

1. **Ingest.** The two laws come from EUR-Lex as PDFs. The code removes page headers, splits the text at article headings (and annexes), and keeps a label like "GDPR - Article 20" on every chunk.
2. **Search.** Chunks are embedded with `BAAI/bge-base-en-v1.5`. A question is searched in two versions, the original and a rewrite in the laws' own vocabulary, keeping each article's better score. The explanatory recitals are left out, so citations point at binding articles.
3. **Gate 1.** If the best score for the original question is below 0.55, the system refuses without calling a language model.
4. **Answer.** `gemini-3.5-flash-lite` (free tier) reads the top five passages and must answer only from them, citing source numbers.
5. **Gate 2.** The model can say "not covered", and my code rejects any citation that wasn't one of the passages it was given.

## What I tried

All retrieval runs use the same 44 development questions with a known answer. One question is worth about 2 points, so differences of one question are noise.

| # | Change | Top 3 | Ranked first | Verdict |
|---|---|---|---|---|
| 1 | Baseline: small embedding model, recitals searched | 61% | 41% | |
| 2 | Search the articles only | 68% | 48% | Helped |
| 3 | Stronger embeddings (bge-base) | 84% | 70% | Helped |
| 4 | Add a re-ranker | 82% | 64% | **Worse, rejected** |
| 5 | Remove amendment markers from the text | 84% | 70% | No change |
| 6 | Merge tiny chunks | 84% | 73% | No change (kept) |
| 7 | 800-character chunks | 86% | 73% | No change |
| 8 | 2,000-character chunks | 86% | 70% | No change |
| 9 | Rewrite the question (prompt with example words) | 93% | 91% | Helped |
| 10 | Rewrite the question (plain prompt) | 93% | 89% | **Adopted** |

Experiment 10 exists because the prompt in 9 had example words I'd picked after seeing which questions failed. The plain version gives the same result, so the gain comes from the idea.

On the answer step I also tested citing source numbers instead of article names (fixed a mislabelled citation), giving the model neighbouring chunks, and a scope-aware prompt. The last two gave no clear gain and cost about 60% more tokens, so the final setup leaves them out.

## How I evaluated it

- **66 questions** I wrote: 52 with a known correct article and 14 the laws shouldn't cover (including other countries' privacy laws). 11 are held-out and were only looked at in totals.
- **A fresh test** of 13 real and 4 off-topic questions, written after the system was finished and run once.
- **20 attacks:** instructions hidden in a real question, prompt extraction, fake authority, forged citations, false premises, requests for help hiding a breach.
- Predictions were written down before each experiment, the answer key was checked against the article text before scoring, and one wrong expected answer was found and fixed that way.

![Red-team results](docs/images/report-redteam.png)

## Honest limits

- **Small test sets.** One question is worth 2 to 8 points.
- **Tuned on the development set.** The rewriting idea came from looking at its failures. On the fresh test it improved ordering (right article first 12/13 vs 8/13) but not top-3 coverage (12/13 either way), and one question got worse.
- **Answers can be incomplete or narrow.** The model sees one chunk of each long article, and giving it neighbouring chunks did not fix this measurably.
- **Over-cautious.** Real questions are sometimes refused, and the refusal says the documents "don't cover this" even when they cover something related. 17 of 20 attacks were refused, including two that could have been answered helpfully.
- **English only.** Arabic questions are refused, although the rewrite step translates them well.
- **Not tested:** multi-turn attacks, poisoned source documents, large-scale red-teaming.
- **A free-tier model.** It changes over time and has daily limits, so a rerun may differ.

## Try it

```
python -m streamlit run src/app.py
```

The app has a live search (type any question and watch the real search and the refusal gate), 92 recorded outputs of the full system with a six-stage trace and the cited article's text, and an About tab. It never calls a language model unless you put a key in `.env`.

![The app](docs/images/app-recorded.png)

## Run it yourself

Tested on Windows with PowerShell, Python 3.11 and 3.13. The first run downloads the embedding model (about 440 MB).

```
git clone https://github.com/PapaMishoo/regulation-rag-checker.git
cd regulation-rag-checker
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src/build_chunks.py --strip-markers --merge-short
python src/embed_chunks.py bge-base
python src/evaluate_rewrite.py my_check
```

The last command needs no API key (it uses the saved question rewrites) and should print **Hit@1 89%, Hit@3 93%, Hit@5 95%, MRR 0.91**, the same numbers I got on two machines.

Other commands:

```
python src/build_report.py          # rebuilds docs/report/index.html from the result files
python src/compare.py               # table of all retrieval runs
python src/compare_answers.py       # table of all answer runs
```

To run the full pipeline (rewriting and answers), create a free key in Google AI Studio, copy `.env.example` to `.env`, and put the key in it. Never commit `.env`. Model names on free tiers change, so if `gemini-3.5-flash-lite` has gone, edit `MODEL` in `src/llm.py`.

| Folder | What's in it |
|---|---|
| `data/` | The two source PDFs and `SOURCES.md` |
| `src/` | Chunking, embedding, search, rewriting, answering, evaluation, the app, the report builder |
| `eval/` | The question files and the saved result of every experiment, answer run, red-team round, and the fresh test |
| `docs/` | The devlog, the generated report page, and images |

## Data and credits
The GDPR (Regulation (EU) 2016/679) and the EU AI Act (Regulation (EU) 2024/1689, consolidated version of 27 July 2026) come from [EUR-Lex](https://eur-lex.europa.eu). © European Union, reused with credit under Commission Decision 2011/833/EU. Consolidated texts are for information only, and the Official Journal text is the authentic legal version. The texts are split into passages for search, and no wording is changed. The MIT licence below covers my code, not these texts.

## How I built it
I built this with AI assistance (Claude) and kept a [devlog](docs/devlog.md) of what I did and why at each step. I set the direction, ran and checked every experiment, wrote down predictions before results, and can explain each part.

## Use
Shared to show my work. All rights reserved.