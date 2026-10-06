from pathlib import Path
import pymupdf

DATA_DIR = Path("data")

for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
    doc = pymupdf.open(pdf_path)

    # Join the text of all pages into one big string
    text = ""
    for page in doc:
        text += page.get_text()

    middle = len(doc) // 2

    print("=" * 60)
    print("File:", pdf_path.name)
    print("Pages:", len(doc))
    print("Characters:", len(text))
    print("Garbled characters:", text.count("\ufffd"))
    print("'Article 5' appears:", text.count("Article 5"), "times")
    print(f"--- Sample from page {middle + 1} ---")
    print(doc[middle].get_text()[:800])