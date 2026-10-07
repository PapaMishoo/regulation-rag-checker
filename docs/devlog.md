# Devlog

## Day 1: Project setup

**What I did:** Installed Git and Python 3.13, created the GitHub repo, copied it to my PC, and set up the project folders (data, src, eval, docs). Then downloaded the EU AI Act and GDPR from EUR-Lex as PDFs and saved them in the data folder, with a sources note.

**Why:** I wanted one place for all my code and notes that I can pick up from any computer, and that also shows my progress step by step. The whole project runs on these two laws, so I wanted the official source and a record of which version I used.

**What I learned:** Git and GitHub keep my work backed up and in sync between machines. Keeping secrets like API keys out of GitHub from the start is much easier than fixing it later. Laws get amended, so I checked the date on each page and used the up-to-date text (the AI Act has a newer consolidated version from July 2026). I also checked that I'm allowed to reuse these documents and noted the credit.

**Checked the PDFs:** Wrote a small script that reads both PDFs. AI Act is 151 pages and GDPR is 88, and both came out as clean text with no broken characters. I noticed each page has a header line and that lines are wrapped mid-sentence, so I'll need to clean those up before splitting the text.

**Chunking results:** The script produced 641 chunks (300 from GDPR, 341 from the AI Act), with no missing article numbers and no leftover page headers. The AI Act has 119 articles instead of 113, and 14 annexes instead of 13, because the 2026 consolidated version includes new articles like 4a and 75a to 75d. A few things I noticed but left alone until I can measure them: some very short chunks and words split across lines by the PDF.

**First working search:** Turned all 641 chunks into embeddings with a small model (all-MiniLM-L6-v2) and built a search that returns the 3 closest articles. Tried 6 questions by hand: for the 5 real questions the right article was in the top 3 every time (rank 1 in four of them, rank 2 for the AI Act fines question), and the off-topic pizza question scored only 0.09. Things I noticed: 291 of 641 chunks are longer than the model can read, so their endings are ignored. The lower-ranked results are often unrelated, and recitals show up a lot. The amendment markers (▼B, ▼M1) from the consolidated text show up in chunks and need cleaning.