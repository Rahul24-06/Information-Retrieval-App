# Information Retrieval System — Assignment 2

End-to-end IR system built with Streamlit for BITS Pilani Assignment 2 (Group 83).

## Features

| Tab | Component |
|-----|-----------|
| A | Document upload, project dataset loader, document viewer |
| B | Preprocessing pipeline, inverted index, preprocessing impact table, stem vs lemma comparison |
| C | **Unified query & retrieve** with preprocessing options and retrieval technique selection |
| D | Phrase query (biword vs positional) |
| E | BST vs B-Tree dictionary search with **search time + retrieval time** benchmark |
| F | Tolerant retrieval (wildcard, spelling, phonetic) |
| G | **Dynamic** inference and discussion driven by experiment results |

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

1. Upload this folder (`IR-assignment-2-complete`) to the BITS Lab portal.
2. Ensure Python 3.9+ is available.
3. Run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```
4. Open the provided portal URL in a browser.
5. Click **Load Project Dataset**, then walk through tabs A → G.

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

## Implementation Notes

- **Preprocessing**: NLTK tokenization, Porter Stemmer, WordNet Lemmatizer
- **BST**: Iterative pure-Python binary search tree
- **B-Tree**: Minimum degree t=3 with split-on-insert
- **Tolerant retrieval**: 2-gram index + regex, Levenshtein edit distance, Soundex
- **Tab G inferences**: Populated from `st.session_state` experiment results

## Authors

- RAHUL KHANNA D — 2025AB05245
- SUKRIT SARKAR — 2025AB05235
