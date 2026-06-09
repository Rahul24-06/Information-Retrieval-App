# Information Retrieval System

## Overview
An end-to-end Information Retrieval system built with Streamlit covering text preprocessing, phrase query processing, dictionary search structures, and tolerant retrieval.

## Features
| Tab | Component |
|-----|-----------|
| A   | Document Upload & Viewing |
| B   | Text Preprocessing + Stemming vs Lemmatization |
| C   | Phrase Query (Biword + Positional Index) |
| D   | Dictionary Search (BST + B-Tree comparison) |
| E   | Tolerant Retrieval (Wildcard, Edit Distance, Soundex) |
| G   | Inference & Discussion |

## Installation

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

## Usage
1. **Tab A** – Upload `.txt` files or click "Load Sample Dataset"
2. **Tab B** – Explore preprocessing pipeline; compare stemming vs lemmatization
3. **Tab C** – Enter a phrase query and compare biword vs positional index results
4. **Tab D** – Search terms in BST and B-Tree; view experimental benchmark table
5. **Tab E** – Try wildcard queries (`ret*iev*`), spelling correction, or phonetic matching
6. **Tab G** – Read full inferences and system architecture

## Implementation Details
- **Preprocessing**: NLTK tokenization, Porter Stemmer, WordNet Lemmatizer
- **Phrase Query**: Custom biword index and positional index with consecutive-position verification
- **BST**: Pure Python recursive BST with comparison counter
- **B-Tree**: Pure Python B-Tree (minimum degree t=3) with split-on-insert
- **Tolerant Retrieval**: K-gram (2-gram) index for wildcards, Levenshtein edit distance for spelling correction, Soundex for phonetic matching

## Dataset
Use the built-in sample dataset (15 documents) or upload your own `.txt` files.
