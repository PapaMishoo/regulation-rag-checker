# Devlog

## Day 1: Project setup

**What I did:** Installed Git and Python 3.13, created the GitHub repo, copied it to my PC, and set up the project folders (data, src, eval, docs). Then downloaded the EU AI Act and GDPR from EUR-Lex as PDFs and saved them in the data folder, with a sources note.

**Why:** I wanted one place for all my code and notes that I can pick up from any computer, and that also shows my progress step by step. The whole project runs on these two laws, so I wanted the official source and a record of which version I used.

**What I learned:** Git and GitHub keep my work backed up and in sync between machines. Keeping secrets like API keys out of GitHub from the start is much easier than fixing it later. Laws get amended, so I checked the date on each page and used the up-to-date text (the AI Act has a newer consolidated version from July 2026). I also checked that I'm allowed to reuse these documents and noted the credit.

**Checked the PDFs:** Wrote a small script that reads both PDFs. AI Act is 151 pages and GDPR is 88, and both came out as clean text with no broken characters. I noticed each page has a header line and that lines are wrapped mid-sentence, so I'll need to clean those up before splitting the text.

**Chunking results:** The script produced 641 chunks (300 from GDPR, 341 from the AI Act), with no missing article numbers and no leftover page headers. The AI Act has 119 articles instead of 113, and 14 annexes instead of 13, because the 2026 consolidated version includes new articles like 4a and 75a to 75d. A few things I noticed but left alone until I can measure them: some very short chunks and words split across lines by the PDF.

**First working search:** Turned all 641 chunks into embeddings with a small model (all-MiniLM-L6-v2) and built a search that returns the 3 closest articles. Tried 6 questions by hand: for the 5 real questions the right article was in the top 3 every time (rank 1 in four of them, rank 2 for the AI Act fines question), and the off-topic pizza question scored only 0.09. Things I noticed: 291 of 641 chunks are longer than the model can read, so their endings are ignored. The lower-ranked results are often unrelated, and recitals show up a lot. The amendment markers (▼B, ▼M1) from the consolidated text show up in chunks and need cleaning.

**Answer key check:** Wrote 33 test questions (27 with a known correct article, 6 that shouldn't match anything). Before running any scores, I printed each expected article's opening to check it. One was wrong: I had listed Article 4a as relevant to AI training, but it's about bias detection data, so I removed it. Checking the key first matters, because a wrong answer in it would quietly ruin every score.

**Baseline scores:** Ran my search on 33 test questions. Right article in the top 3 for 67% (44% at rank 1), with an average ranking score of 0.55. It did best on questions worded like the law (80%) and worst on questions needing several articles (50%). Biggest problem: the explanatory recitals often rank above the actual articles, 4 of the 9 misses. Another: some AI Act questions pull neighbouring articles with high scores, so a high score doesn't mean a correct answer. My test set is small, so I won't read much into changes of one question.

**Experiment 1: leave out the recitals.** I predicted Hit@3 would rise from 67% to about 75-80%. It went to 74% (Hit@1 44% to 56%, Hit@5 70% to 85%, average ranking score 0.55 to 0.66). Nothing that worked before broke. The recitals explain the law but aren't binding, and this tool cites articles, so it makes sense to search only articles and annexes. What it didn't fix: everyday-language questions still miss (like "can I move my data to another service?"), and the AI Act's high-risk articles get confused with each other. My test set is small, so I'm treating a one-question change as noise.

**Cutoff test:** After switching models, my old "I don't know" cutoff stopped working because the new model gives higher scores in general. Re-testing, a cutoff of 0.55 keeps all my real questions and refuses 4 of 6 off-topic ones. A stricter 0.6 refuses 5 of 6 but also wrongly refuses 3 real questions. I'll use 0.55 for now, but only 6 of my questions are off-topic, so I'll add more before trusting it.