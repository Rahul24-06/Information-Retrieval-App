# Information Retrieval System — Assignment 1

End-to-end IR system built with Streamlit for BITS Pilani Assignment 1 (Group 83).


## Installation

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

## BITS Lab Portal Deployment


1. Ensure Python 3.9+ is available.
2. Run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```
3. Open the provided portal URL in a browser.
4. Click **Load Project Dataset**, then walk through tabs A → G.

## Recommended Demo Flow

1. **Tab A** — Click **Load Project Dataset** (8 news documents)
2. **Tab B** — Review preprocessing pipeline; note preprocessing impact table and stem/lemma conclusion
3. **Tab C** — Query `information retrieval` with Boolean AND; try `ret*` wildcard
4. **Tab D** — Phrase query `microsoft investigating` (biword vs positional)
5. **Tab E** — Review fixed-query BST vs B-Tree benchmark table
6. **Tab F** — Try `retreival` spelling correction
7. **Tab G** — Read data-driven inferences (run other tabs first)

## Dataset

- `dataset/` — 8 text documents (news articles with hyphens like anti-spyware, e-mail)
