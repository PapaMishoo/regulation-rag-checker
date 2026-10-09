import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "processed"
MAX_CHARS = 1500  # default chunk size
MIN_CHARS = 300   # with --merge-short, a last piece shorter than this joins the piece before it

SOURCES = {
    "GDPR": "gdpr_2016_679_en.pdf",
    "EU AI Act": "ai_act_2024_1689_consolidated_2026-07-27_en.pdf",
}

# Page headers and footers that are not law text
PART = r"(L\s*\d+/\d+|\d{1,2}\.\d{1,2}\.\d{4}|EN)"
NOISE_PATTERNS = [
    re.compile(r"0\d{4}R\d{4}\s*—\s*EN\s*—.*"),  # AI Act page header
    re.compile(rf"({PART}\s+)*Official Journal of the European Union(\s+{PART})*"),  # GDPR header
    re.compile(r"L\s*\d+/\d+"),
    re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}"),
    re.compile(r"EN"),
]

ARTICLE_RE = re.compile(r"^Article (\d+)([a-z]?)$")
ANNEX_RE = re.compile(r"^ANNEX ([IVXLC]+)$")
STRUCTURE_RE = re.compile(r"^(CHAPTER [IVXLC]+|SECTION \d+)$")
MARKER_RE = re.compile(r"[▼►][A-Z]\d*|◄")  # EUR-Lex amendment markers like ▼B and ▼M1


def is_noise(line):
    return any(p.fullmatch(line) for p in NOISE_PATTERNS)


def load_lines(pdf_path):
    """Read a PDF and return its text as a list of clean lines."""
    doc = pymupdf.open(pdf_path)
    lines = []
    for page in doc:
        for line in page.get_text().splitlines():
            line = line.strip()
            if line and not is_noise(line):
                lines.append(line)
    return lines


def is_next_article(num, suffix, last_num, last_suffix):
    # Articles run 1, 2, 3... (sometimes 4a, 4b). This avoids mistaking a
    # cross-reference that happens to sit alone on a line for a new article.
    return num == last_num + 1 or (num == last_num and suffix > last_suffix)


def split_sections(lines):
    """Cut the text into sections: preamble, articles, annexes."""
    sections = []
    current = {"kind": "Preamble", "number": "", "lines": []}
    last_num, last_suffix = 0, ""
    for line in lines:
        article = ARTICLE_RE.match(line)
        annex = ANNEX_RE.match(line)
        if article and is_next_article(int(article.group(1)), article.group(2), last_num, last_suffix):
            sections.append(current)
            last_num, last_suffix = int(article.group(1)), article.group(2)
            current = {"kind": "Article", "number": article.group(1) + article.group(2), "lines": []}
        elif annex:
            sections.append(current)
            current = {"kind": "Annex", "number": annex.group(1), "lines": []}
        elif STRUCTURE_RE.match(line):
            continue  # skip "CHAPTER II" and "SECTION 1" lines
        else:
            current["lines"].append(line)
    sections.append(current)
    return sections


def join_lines(lines, strip_markers=False):
    """Glue the wrapped lines back into normal flowing text."""
    text = re.sub(r"\s+", " ", " ".join(lines))
    if strip_markers:
        text = re.sub(r"\s+", " ", MARKER_RE.sub("", text))
    return text.strip()


def split_text(text, max_chars, merge_short=False):
    """Break long text into chunks of up to max_chars, cutting between sentences."""
    pieces = re.split(r"(?<=[.;:])\s+", text)
    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    if merge_short and len(chunks) > 1 and len(chunks[-1]) < MIN_CHARS:
        last = chunks.pop()
        chunks[-1] = f"{chunks[-1]} {last}"
    return chunks


def build_chunks(max_chars=MAX_CHARS, strip_markers=False, merge_short=False):
    all_chunks = []
    for source, filename in SOURCES.items():
        lines = load_lines(DATA_DIR / filename)
        sections = split_sections(lines)

        chunks = []
        for section in sections:
            text = join_lines(section["lines"], strip_markers)
            if not text:
                continue
            if section["kind"] == "Preamble":
                label = f"{source} - Preamble"
            else:
                label = f"{source} - {section['kind']} {section['number']}"
            for i, part in enumerate(split_text(text, max_chars, merge_short), start=1):
                chunks.append({
                    "id": f"{label} #{i}",
                    "label": label,
                    "source": source,
                    "text": part,
                })

        report(source, lines, sections, chunks)
        all_chunks.extend(chunks)
    return all_chunks


def report(source, lines, sections, chunks):
    articles = [s for s in sections if s["kind"] == "Article"]
    annexes = [s for s in sections if s["kind"] == "Annex"]
    lengths = [len(c["text"]) for c in chunks]
    numbers = {int(re.match(r"\d+", s["number"]).group()) for s in articles}
    missing = [n for n in range(1, max(numbers) + 1) if n not in numbers] if numbers else []

    print("=" * 60)
    print(source)
    print("Articles found:", len(articles))
    print("Annexes found:", len(annexes))
    print("Missing article numbers:", missing if missing else "none")
    suffixed = [s["number"] for s in articles if not s["number"].isdigit()]
    print("Articles with a letter (like 4a):", suffixed if suffixed else "none")
    print("Annex numbers:", [s["number"] for s in annexes])
    print("Chunks:", len(chunks))
    print("Chunk length (min / average / max):", min(lengths), sum(lengths) // len(lengths), max(lengths))
    print("Amendment markers left in chunks:", sum(c["text"].count("▼") + c["text"].count("►") for c in chunks))
    print("Most repeated lines (leftover page headers show up here):")
    long_lines = [line for line in lines if len(line) > 6]
    for line, count in Counter(long_lines).most_common(5):
        print(f"  {count}x  {line[:70]}")


def show_sample(chunks, label):
    for c in chunks:
        if c["label"] == label:
            print("=" * 60)
            print("Sample chunk:", c["id"])
            print(c["text"][:700])
            return
    print("Sample not found:", label)


if __name__ == "__main__":
    flags = sys.argv[1:]
    max_chars = MAX_CHARS
    for flag in flags:
        if flag.startswith("--max-chars="):
            max_chars = int(flag.split("=", 1)[1])
    options = {
        "max_chars": max_chars,
        "strip_markers": "--strip-markers" in flags,
        "merge_short": "--merge-short" in flags,
    }

    chunks = build_chunks(**options)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "chunks.jsonl", "w", encoding="utf-8") as out:
        for c in chunks:
            out.write(json.dumps(c, ensure_ascii=False) + "\n")
    (OUT_DIR / "chunks_config.json").write_text(json.dumps(options, indent=2), encoding="utf-8")

    print("=" * 60)
    print("Chunk settings:", options)
    print("Total chunks saved:", len(chunks))
    show_sample(chunks, "GDPR - Article 20")
    show_sample(chunks, "EU AI Act - Article 5")