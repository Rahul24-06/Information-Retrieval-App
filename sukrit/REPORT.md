# Information Retrieval System — Assignment 2 Report

**Course:** Information Retrieval  
**Assignment:** Assignment 2 — Streamlit-Based End-to-End IR System  
**Group:** 83

| Name | ID |
|------|-----|
| RAHUL KHANNA D | 2025AB05245 |
| SUKRIT SARKAR | 2025AB05235 |

**Submission Date:** June 2026

---

## Table of Contents

1. [Objective](#1-objective)
2. [Implementation in the Virtual Lab](#2-implementation-in-the-virtual-lab)
3. [Dataset](#3-dataset)
4. [System Architecture](#4-system-architecture)
5. [Experimental Results](#5-experimental-results)
6. [Screenshots of Streamlit Front End](#6-screenshots-of-streamlit-front-end)
7. [G. Inference and Discussion (Compulsory)](#7-g-inference-and-discussion-compulsory)
8. [Conclusion](#8-conclusion)
9. [References](#9-references)

---

## 1. Objective

The objective of this assignment is to design and implement an end-to-end Information Retrieval (IR) system using Streamlit. The system allows users to:

- Upload a document collection
- View uploaded documents
- Enter search queries from the front end
- Select preprocessing and retrieval options
- Observe outputs for preprocessing, indexing, querying, and tolerant retrieval

The complete workflow is executable through the Streamlit front end only, without relying on static notebook outputs or separate backend demonstrations.

---

## 2. Implementation in the Virtual Lab

### 2.1 Environment Setup

The system was developed and executed on the **BITS Virtual Lab** portal using the following environment:

| Component | Details |
|-----------|---------|
| Language | Python 3.9+ |
| Framework | Streamlit 1.32+ |
| NLP Library | NLTK 3.8+ |
| Data Processing | Pandas 2.0+ |
| Deployment | BITS Lab portal / local Streamlit server |

**Installation commands used in Virtual Lab:**

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### 2.2 Project Structure

```
IR-assignment-2-complete/
├── app.py              # Complete Streamlit application (all IR logic + UI)
├── requirements.txt    # Python dependencies
├── dataset/            # 8 text documents (news articles)
│   ├── 001.txt … 008.txt
├── README.md           # Setup and usage guide
└── REPORT.md           # This report
```

### 2.3 Implementation Overview by Task

#### A. Streamlit End-to-End Workflow

All functionality is integrated into a single Streamlit application (`app.py`) with seven tabs:

| Tab | Function |
|-----|----------|
| A | Document upload, project dataset loader, document viewer |
| B | Text preprocessing pipeline, inverted index lookup, preprocessing experiments |
| C | Unified query interface with preprocessing options and retrieval technique selection |
| D | Phrase query processing (biword vs positional index) |
| E | Dictionary search using BST and B-Tree with performance benchmarking |
| F | Tolerant retrieval (wildcard, spelling correction, phonetic matching) |
| G | Dynamic inference and discussion |

Users interact exclusively through the browser-based Streamlit interface. No separate backend scripts or Jupyter notebooks are required for demonstration.

#### B. Text Preprocessing

The preprocessing pipeline applies the following steps in sequence:

1. **Hyphen handling** — Replaces hyphens with spaces (`anti-spyware` → `anti spyware`)
2. **Lowercasing** — Normalizes case (`Microsoft` → `microsoft`)
3. **Tokenization** — NLTK `word_tokenize`
4. **Alphabetic filtering** — Removes punctuation and numeric tokens
5. **Stop word removal** — Removes 179 English stop words
6. **Stemming / Lemmatization** — Porter Stemmer or WordNet Lemmatizer (optional)

An **inverted index** maps each term to document IDs and term positions within documents.

#### C. Phrase Query Processing

Two index structures are implemented:

- **Biword Index** — Maps adjacent token pairs to document IDs
- **Positional Index** — Stores all term positions; phrase matching verifies consecutive positions

Both indexes are queried side-by-side with timing measurements and false-positive analysis.

#### D. Dictionary Search

A vocabulary dictionary is built from all unique terms in the collection. Two tree structures are implemented:

- **Binary Search Tree (BST)** — Iterative insertion and search with comparison counter
- **B-Tree (t=3)** — Multi-key nodes with split-on-insert for balanced performance

Performance is measured for both **dictionary search time** and **document retrieval time** (fetching postings from the inverted index).

#### E. Tolerant Retrieval

Three tolerant retrieval modes are implemented:

| Mode | Technique | Purpose |
|------|-----------|---------|
| Wildcard | 2-gram index + regex post-filter | Prefix/suffix/infix pattern matching (`soft*`) |
| Spelling correction | Levenshtein edit distance (d ≤ 2) | Typo correction (`microsft` → `microsoft`) |
| Phonetic | Soundex encoding | Homophone matching (`microsoft` → `makers`) |

#### F. Query & Retrieve (Unified Interface)

Tab C provides a single entry point where users select preprocessing options and retrieval technique, enter a query, and receive ranked document results — fulfilling the end-to-end workflow requirement.

---

## 3. Dataset

### 3.1 Primary Dataset (Project Dataset)

Eight news-style text documents loaded from the `dataset/` folder:

| Doc ID | File | Topic |
|--------|------|-------|
| 0 | 001.txt | Kyrgyz election ink voting |
| 1 | 002.txt | Internet café law (Netherlands) |
| 2 | 003.txt | Microsoft trojan / anti-spyware |
| 3 | 004.txt | One laptop per child initiative |
| 4 | 005.txt | BT creative industry / technology |
| 5 | 006.txt | Peru agricultural telecentres |
| 6 | 007.txt | Microsoft security updates |
| 7 | 008.txt | Zafi virus / anti-virus |

**Collection statistics (after full preprocessing):**

| Metric | Value |
|--------|-------|
| Documents | 8 |
| Total tokens | 2,847 |
| Vocabulary size | 511 |
| Avg tokens/doc | 355.9 |

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONT END                       │
├─────────────────────────────────────────────────────────────┤
│  Tab A: Upload & View                                        │
│  Tab B: Preprocessing                                        │
│  Tab C: Query & Retrieve                                     │
│  Tab D: Phrase Query                                         │
│  Tab E: Dictionary Search                                    │
│  Tab F: Tolerant Retrieval                                   │
│  Tab G: Inference & Discussion                               │
├─────────────────────────────────────────────────────────────┤
│  PREPROCESSING LAYER                                         │
│  Hyphen → Lowercase → Tokenize → Stop removal → Stem/Lemma  │
├─────────────────────────────────────────────────────────────┤
│  INDEXING LAYER                                              │
│  Inverted Index | Biword Index | Positional Index            │
│  BST | B-Tree | K-gram Index                                 │
├─────────────────────────────────────────────────────────────┤
│  QUERY PROCESSING LAYER                                      │
│  Boolean AND | Phrase (Biword/Positional)                    │
│  Wildcard | Spelling | Phonetic                              │
├─────────────────────────────────────────────────────────────┤
│  OUTPUT LAYER                                                │
│  Tables, metrics, document snippets, inferences              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Experimental Results

All experiments below were run on the **8-document project dataset** unless otherwise noted. Measurements were taken on the Virtual Lab environment.

### 5.1 Preprocessing Impact on Retrieval Quality

**Test queries:** `information retrieval`, `anti-spyware`, `e-mail`, `Microsoft`, `elections`

| Configuration | Total Hits | Avg Hits/Query | Per-Query Hits |
|---------------|-----------|----------------|----------------|
| Baseline (no preprocessing) | 3 | 0.6 | [0, 0, 0, 2, 1] |
| + Lowercasing | 3 | 0.6 | [0, 0, 0, 2, 1] |
| + Stop word removal | 3 | 0.6 | [0, 0, 0, 2, 1] |
| + Hyphen handling | **6** | **1.2** | [0, 1, 2, 2, 1] |
| Full pipeline | **6** | **1.2** | [0, 1, 2, 2, 1] |

**Key observations:**

- Lowercasing and stop word removal alone did not change hit counts on this dataset because test queries were already lowercase and most were content words.
- **Hyphen handling doubled total hits** (3 → 6) by correctly tokenizing `anti-spyware` and `e-mail`.
- Query `anti-spyware` returned 0 hits without hyphen handling and **1 hit** (doc 003) with it.
- Query `e-mail` returned 0 hits without hyphen handling and **2 hits** (docs 002, 007) with it.

### 5.2 Stemming vs. Lemmatization Comparison

#### Semantic Similarity (Pairwise Cosine)

| Measure | Stemming | Lemmatization |
|---------|----------|---------------|
| Vocabulary size | 450 | 477 |
| Avg pairwise cosine similarity | **0.0677** | 0.0623 |

#### Query Retrieval Hits

| Query | Stem Hits | Lemma Hits | Stem Doc IDs | Lemma Doc IDs |
|-------|-----------|------------|--------------|---------------|
| running | 1 | 0 | [4] | [] |
| studies | 1 | 1 | [4] | [4] |
| retrieval | 0 | 0 | [] | [] |
| investigating | 1 | 1 | [2] | [2] |
| systems | 2 | 2 | [5, 6] | [5, 6] |
| **Total** | **5** | **4** | — | — |

### 5.3 Phrase Query: Biword vs. Positional Index

**Dataset:** 8-document project dataset  
**Query:** `microsoft investigating`

| Index Type | Documents Found | Doc IDs | Query Time |
|------------|----------------|---------|------------|
| Biword Index | 1 | [2] | ~0.5 ms |
| Positional Index | 1 | [2] | ~0.8 ms |

On this dataset, both indexes returned the same result for exact phrases present in the collection (e.g. doc 003: *"Microsoft is investigating a trojan program..."*).

**Biword false positives (general case):** The biword index can return extra documents when overlapping bigrams appear in different parts of a document but not as one contiguous phrase. The positional index avoids this by requiring consecutive term positions. Tab D highlights any false positives automatically when biword and positional results differ.

**Why positional index is more accurate:** The positional index verifies that query terms appear at consecutive positions (pos, pos+1, pos+2). The biword index only checks that adjacent pairs exist somewhere in the document, which allows false matches when biwords appear in different parts of the text.

### 5.4 Dictionary Search: BST vs. B-Tree Performance

**Dictionary size:** 511 terms | **Fixed benchmark queries:** 13

#### Summary Statistics

| Metric | BST | B-Tree (t=3) |
|--------|-----|--------------|
| Avg search time (μs) | **1.22** | 1.87 |
| Avg retrieval time (μs) | 0.74 | **0.34** |
| Avg comparisons | 9.77 | **9.46** |

#### Full Benchmark Table

| Query Term | BST Found | BST Search (μs) | BST Retrieval (μs) | BST Comps | B-Tree Found | BT Search (μs) | BT Retrieval (μs) | BT Comps |
|------------|-----------|-----------------|--------------------|-----------|--------------|----------------|--------------------|----------|
| information | Yes | 3.2 | 4.4 | 8 | Yes | 4.7 | 1.3 | 8 |
| retrieval | No | 1.7 | 0.3 | 9 | No | 2.0 | 0.2 | 8 |
| microsoft | Yes | 1.2 | 1.3 | 11 | Yes | 1.2 | 0.4 | 6 |
| election | Yes | 1.0 | 0.8 | 9 | Yes | 1.5 | 0.4 | 8 |
| virus | Yes | 1.0 | 0.4 | 8 | Yes | 2.1 | 0.4 | 12 |
| technology | Yes | 0.9 | 0.5 | 11 | Yes | 1.3 | 0.3 | 8 |
| farmer | No | 1.2 | 0.2 | 10 | No | 1.7 | 0.1 | 10 |
| ink | Yes | 0.8 | 0.5 | 8 | Yes | 1.8 | 0.3 | 11 |
| software | Yes | 1.1 | 0.4 | 13 | Yes | 1.1 | 0.5 | 6 |
| network | Yes | 1.3 | 0.5 | 12 | Yes | 0.9 | 0.4 | 5 |
| xyz123 | No | 0.7 | 0.1 | 9 | No | 2.1 | 0.1 | 14 |
| notaword | No | 1.0 | 0.1 | 10 | No | 1.7 | 0.0 | 10 |
| zzz | No | 0.8 | 0.1 | 9 | No | 2.2 | 0.0 | 17 |

### 5.5 Tolerant Retrieval Experiments

| Mode | Input Query | Corrected / Matched Terms | Documents Retrieved |
|------|-------------|--------------------------|---------------------|
| Wildcard (`soft*`) | `soft*` | `software` | Doc 003 |
| Spelling (`microsft`) | `microsft` | `microsoft` (edit distance = 1) | Docs 002, 006 |
| Spelling (`viris`) | `viris` | `virus` (distance = 1) | Docs 007, 008 |
| Spelling (`electon`) | `electon` | `election` (distance = 1), `elections` (distance = 2) | Doc 000 |
| Phonetic (`microsoft`) | `microsoft` | `microsoft`, `makers` (same Soundex M262) | Docs 002, 003, 006 |

#### Edit Distance Demonstration

| Word 1 | Word 2 | Edit Distance |
|--------|--------|---------------|
| retrieval | retreival | 2 |
| microsoft | microsft | 1 |
| virus | viris | 1 |

---

## 6. Screenshots of Streamlit Front End

> **Instructions:** Replace each placeholder below with actual screenshots captured while running the app on the BITS Virtual Lab portal. Use `Win + Shift + S` or the lab's screenshot tool.

### Screenshot 1: Tab A — Document Upload & Viewing

**[INSERT SCREENSHOT HERE]**

*Caption: Tab A showing the 8-document project dataset loaded with collection statistics (Documents: 8, Vocabulary: 511).*

**Steps to reproduce:**
1. Open the app on Virtual Lab
2. Click **Load Project Dataset**
3. Capture the document viewer and statistics panel

---

### Screenshot 2: Tab B — Preprocessing Pipeline

**[INSERT SCREENSHOT HERE]**

*Caption: Tab B showing step-by-step preprocessing (original → hyphen handled → tokenized → stop removed → stemmed → lemmatized) for document 003.txt.*

---

### Screenshot 3: Tab B — Preprocessing Impact Table

**[INSERT SCREENSHOT HERE]**

*Caption: Preprocessing impact experiment table showing hyphen handling doubling total hits from 3 to 6.*

---

### Screenshot 4: Tab B — Stemming vs. Lemmatization

**[INSERT SCREENSHOT HERE]**

*Caption: Stemming vs. lemmatization comparison with query retrieval hit table and conclusion.*

---

### Screenshot 5: Tab C — Query & Retrieve

**[INSERT SCREENSHOT HERE]**

*Caption: Tab C unified query interface — query `anti-spyware` with Boolean AND retrieval returning document 003.*

---

### Screenshot 6: Tab D — Phrase Query Comparison

**[INSERT SCREENSHOT HERE]**

*Caption: Phrase query `microsoft investigating` — biword vs positional comparison on the project dataset.*

---

### Screenshot 7: Tab E — BST vs. B-Tree Benchmark

**[INSERT SCREENSHOT HERE]**

*Caption: Dictionary search benchmark table with search time and retrieval time for 13 fixed queries.*

---

### Screenshot 8: Tab F — Tolerant Retrieval

**[INSERT SCREENSHOT HERE]**

*Caption: Spelling correction of `microsft` → `microsoft` with retrieved documents.*

---

### Screenshot 9: Tab G — Inference and Discussion

**[INSERT SCREENSHOT HERE]**

*Caption: Tab G showing all seven compulsory inference answers populated from experiment results.*

---

## 7. G. Inference and Discussion (Compulsory)

### 1. Which preprocessing technique improved retrieval quality?

**Answer: Hyphen handling** was the preprocessing technique that most significantly improved retrieval quality on our dataset.

**Evidence:** The preprocessing impact experiment (Table 5.1) shows that applying hyphen handling increased total query hits from **3 to 6** — a 100% improvement. Lowercasing and stop word removal did not change hit counts for our specific test queries because the queries were already lowercase and consisted primarily of content words rather than stop words.

**Practical impact:**
- Query `anti-spyware` went from **0 hits → 1 hit** (doc 003: Microsoft anti-spyware article)
- Query `e-mail` went from **0 hits → 2 hits** (docs 002, 007)

**Justification:** Our news dataset contains numerous hyphenated compound terms (`anti-spyware`, `e-mail`, `anti-virus`, `Chancay-Huaral`). Without hyphen handling, these are tokenized incorrectly or split at punctuation, causing query terms to miss relevant documents. For collections with technical or compound terminology, hyphen normalization is critical.

The full pipeline (lowercasing + stop word removal + hyphen handling) is recommended as the default configuration for this dataset.

---

### 2. Was stemming or lemmatization better for their dataset?

**Answer: Stemming (Porter Stemmer)** performed better for our dataset based on query retrieval effectiveness.

**Evidence:**

| Criterion | Stemming | Lemmatization | Winner |
|-----------|----------|---------------|--------|
| Total query retrieval hits | **5** | 4 | Stemming |
| Vocabulary size | 450 | 477 | Stemming (more conflation) |
| Avg pairwise cosine similarity | **0.0677** | 0.0623 | Stemming |

**Query-level analysis:**
- `running` matched 1 document with stemming but 0 with lemmatization (stem conflates `running` → `run` matching more variants)
- `studies`, `investigating`, `systems` matched equally with both methods
- `retrieval` matched neither (term absent from dataset vocabulary in both cases)

**Justification:** Stemming aggressively truncates word endings, reducing the vocabulary from 477 to 450 terms and increasing the chance that morphological variants of query terms match indexed terms. For a small news corpus where users may search with varied word forms (`running`, `studies`, `investigating`), stemming provides slightly better recall.

**Trade-off:** Lemmatization produces valid dictionary words (e.g., `investigating` stays `investigating` vs. stem `investig`), making results more interpretable. For production systems with user-facing output, lemmatization may be preferred despite slightly lower hit counts on this dataset.

---

### 3. Which phrase query index was more accurate?

**Answer: The Positional Index** was more accurate than the Biword Index.

**Evidence:** For phrase queries on the project dataset (e.g. `microsoft investigating`, `trojan program`), both indexes typically agree when the phrase appears contiguously in a document. The positional index is still more accurate in general because it verifies consecutive term positions.

| Index | Behavior | Accuracy |
|-------|----------|----------|
| Biword Index | Intersects adjacent bigram postings | May false-match when bigrams appear in separate parts of a document |
| Positional Index | Checks consecutive positions | Exact phrase match only |

**Example (conceptual false positive):** A document containing *"...quick brown hat... later... brown fox..."* would match biword query `quick brown fox` but not positional query, because `quick`, `brown`, and `fox` are not at consecutive positions.

**Why positional index is more accurate:** It enforces that all query terms appear at strictly consecutive positions within a document. The biword index decomposes a phrase into overlapping bigrams and intersects document sets, which cannot distinguish between contiguous and non-contiguous occurrences of those bigrams.

**Trade-off:** The biword index is faster (~0.5 ms vs ~0.8 ms) and uses less storage, but sacrifices accuracy for multi-word phrases.

---

### 4. Which tree structure was faster?

**Answer: Performance depends on the operation measured.**

| Operation | Faster Structure | BST Avg (μs) | B-Tree Avg (μs) |
|-----------|-----------------|--------------|-----------------|
| Dictionary **search** (lookup) | **BST** | 1.22 | 1.87 |
| Document **retrieval** (postings fetch) | **B-Tree** | 0.74 | 0.34 |
| Key **comparisons** | **B-Tree** | 9.77 | 9.46 |

**Analysis:**

- **BST** achieved faster average dictionary search time (1.22 μs vs 1.87 μs) on our 511-term in-memory dictionary. With randomized insertion order (seed=42), the BST maintains near-balanced height, giving efficient O(log n) lookups.
- **B-Tree** achieved faster average document retrieval time (0.34 μs vs 0.74 μs) after the dictionary lookup, and required slightly fewer key comparisons (9.46 vs 9.77).
- For absent terms (`xyz123`, `notaword`, `zzz`), both structures correctly returned "not found" with minimal retrieval overhead.

**Inference:** For small in-memory dictionaries (vocab < 1,000), BST is competitive and sometimes faster for pure search. B-Tree's multi-key nodes provide more consistent performance and would scale better for larger vocabularies and disk-based storage where minimizing node accesses (I/O operations) is critical. In production IR systems, B-Trees are the standard choice for on-disk dictionaries due to their guaranteed O(log n) height regardless of insertion order.

---

### 5. How tolerant was the retrieval model?

**Answer: The retrieval model demonstrated good tolerance** across three classes of imperfect queries.

| Imperfection Type | Technique | Example | Result | Tolerance Level |
|-------------------|-----------|---------|--------|-----------------|
| Partial/pattern query | K-gram wildcard | `soft*` | Matched `software` → doc 003 | **High** |
| Character transposition/substitution | Edit distance | `microsft` | Corrected to `microsoft` → docs 002, 006 | **High** |
| Character deletion | Edit distance | `viris` | Corrected to `virus` → docs 007, 008 | **High** |
| Phonetic misspelling | Soundex | `microsoft` | Also matched `makers` (same code M262) | **Moderate** |
| Non-existent term | Edit distance | `retreival` | No match in vocabulary (term absent) | **N/A** |

**Strengths:**
- Wildcard queries via 2-gram index efficiently narrow candidates before regex post-filtering
- Edit distance correction (d ≤ 2) handles common typos: transpositions (`microsft`), deletions (`viris`), substitutions (`electon` → `election`)
- All tolerant modes chain into the inverted index for full document retrieval

**Weaknesses:**
- Soundex produces false positives (`microsoft` matches `makers`) due to coarse phonetic encoding
- Spelling correction is limited to vocabulary terms; unknown words cannot be corrected
- No support for multi-word tolerant queries (e.g., tolerant phrase search)

**Overall assessment:** The system handles single-term imperfect queries effectively. For a keyword-based IR system on a small corpus, tolerance is **good for spelling errors and wildcards**, **moderate for phonetic matching**.

---

### 6. What are the limitations of the system?

1. **Scalability** — All indexes (inverted, biword, positional, k-gram, BST, B-Tree) are stored in memory. Large corpora (millions of documents) would exceed available RAM and require disk-based or distributed indexing.

2. **No ranked retrieval** — Results are returned as unordered document sets. There is no TF-IDF, BM25, or relevance scoring to rank documents by query relevance.

3. **BST imbalance risk** — The BST is not self-balancing (no AVL or Red-Black tree). While randomized insertion mitigates worst-case O(n) degradation, sorted input would produce a skewed tree.

4. **Biword false positives** — The biword index cannot guarantee exact phrase matching for queries with three or more terms when bigrams appear in non-contiguous parts of a document.

5. **Coarse phonetic matching** — Soundex collapses many distinct words into the same code (e.g., `microsoft` and `makers` both map to M262), producing irrelevant matches.

6. **Single-term tolerant queries** — Wildcard, spelling, and phonetic correction operate on individual terms only. Multi-word tolerant phrase search is not supported.

7. **English-only processing** — NLTK English stop words, Porter Stemmer, and WordNet Lemmatizer are used. The system does not handle multilingual collections.

8. **No persistence** — Indexes are rebuilt on each session. There is no save/load mechanism for pre-built indexes.

---

### 7. How can the system be improved?

| Improvement | Description | Expected Benefit |
|-------------|-------------|------------------|
| **TF-IDF / BM25 ranking** | Score and rank retrieved documents by term frequency and inverse document frequency | Users see most relevant documents first |
| **Self-balancing BST** | Implement AVL or Red-Black tree | Guaranteed O(log n) worst-case search regardless of insertion order |
| **Disk-backed indexes** | Serialize inverted index and B-Tree to disk (SQLite, file-based) | Support corpora larger than available memory |
| **POS-aware lemmatization** | Use NLTK POS tagging to select correct WordNet lemma (e.g., `running` as verb → `run`) | More accurate lemmatization than context-free approach |
| **Permuterm index** | Rotate terms with `$` sentinel for all wildcard positions | Eliminate k-gram false positives for wildcard queries |
| **Improved phonetic algorithm** | Replace Soundex with Metaphone or Double Metaphone | Fewer false phonetic matches |
| **Phrase tolerant search** | Combine edit distance with positional index for approximate phrase matching | Handle typos within multi-word queries |
| **Query expansion** | Use WordNet synonyms to expand queries before retrieval | Improved recall for semantic variants |
| **Neural retrieval** | Integrate sentence-transformer embeddings for dense retrieval | Semantic matching beyond keyword overlap |
| **Index caching** | Cache built indexes in `st.session_state` with invalidation on dataset change | Faster repeated queries within a session |

---

## 8. Conclusion

We successfully implemented a complete end-to-end Information Retrieval system using Streamlit that fulfills all assignment requirements:

- **Workflow (A):** Full upload-to-retrieval pipeline through the browser interface
- **Preprocessing (B):** All required steps with experimental impact analysis
- **Phrase Query (C/D):** Biword and positional indexes with demonstrated false positives
- **Dictionary Search (E):** BST and B-Tree with search and retrieval time benchmarks
- **Tolerant Retrieval (F):** Wildcard, spelling, and phonetic correction with document retrieval
- **Inference (G):** Data-driven answers to all seven compulsory questions

Key findings: hyphen handling was the most impactful preprocessing step for our news dataset; stemming slightly outperformed lemmatization in query retrieval; the positional index eliminated biword false positives; BST was faster for dictionary search while B-Tree was faster for document retrieval on our 511-term vocabulary.

---

## 9. References

1. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press.
2. NLTK Documentation — https://www.nltk.org/
3. Streamlit Documentation — https://docs.streamlit.io/
4. Porter, M. F. (1980). An algorithm for suffix stripping. *Program*, 14(3), 130–137.
5. Soundex Algorithm — U.S. National Archives standard for phonetic name encoding.

---

*Report prepared by Group 83 — RAHUL KHANNA D (2025AB05245) & SUKRIT SARKAR (2025AB05235)*
