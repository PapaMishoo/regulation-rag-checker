import json
import re
import time
from pathlib import Path

import numpy as np

import llm

PAUSE = 5  # seconds between uncached model calls, to stay inside the free tier's limits
import os

VARIANT = os.getenv("REWRITE_PROMPT", "plain")  # "plain" (adopted) has no example words; "vocab" lists a few example terms

PROMPTS = {
    "vocab": """You help a search engine find articles of the GDPR and the EU AI Act.
Rewrite the user's question as a short search query written in the formal vocabulary these regulations use (for example: data subject, controller, processor, supervisory authority, provider, deployer, high-risk AI system, placing on the market).
Keep the meaning exactly the same. Do not answer the question. Do not mention article numbers.
Reply with JSON only, in exactly this shape:
{"rewritten": "..."}""",
    "plain": """You help a search engine find articles of the GDPR and the EU AI Act.
Rewrite the user's question as a short search query written in the formal legal vocabulary these regulations use.
Keep the meaning exactly the same. Do not answer the question. Do not mention article numbers.
Reply with JSON only, in exactly this shape:
{"rewritten": "..."}""",
}
SYSTEM_PROMPT = PROMPTS[VARIANT]
CACHE_PATH = Path("eval") / (
    f"rewrites_{llm.MODEL}.json" if VARIANT == "vocab" else f"rewrites_{llm.MODEL}_{VARIANT}.json")

_cache = None


def get_rewrite(question):
    """Return the rewritten question, calling the model only if it hasn't been done before."""
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    if question not in _cache:
        raw, _tokens_in, _tokens_out = llm.generate(SYSTEM_PROMPT, question, max_tokens=300)
        rewritten = question  # fall back to the original if the reply can't be read
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                rewritten = json.loads(match.group(0)).get("rewritten") or question
            except json.JSONDecodeError:
                pass
        _cache[question] = rewritten
        CACHE_PATH.write_text(json.dumps(_cache, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(PAUSE)
    return _cache[question]


def search_with_rewrite(question, rewritten, embedder, chunks, embeddings, top_k):
    """Search with both versions of the question and keep each chunk's better score.
    Also returns the best score of the ORIGINAL question, which the refusal cutoff uses."""
    vectors = np.stack([embedder.encode_query(question), embedder.encode_query(rewritten)])
    scores = embeddings @ vectors.T            # one column per version of the question
    best = scores.max(axis=1)
    order = np.argsort(best)[::-1][:top_k]
    original_top = float(scores[:, 0].max())
    return [(chunks[i], float(best[i])) for i in order], original_top