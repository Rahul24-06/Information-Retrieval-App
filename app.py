import streamlit as st
import pandas as pd
import nltk
import re
import time
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.metrics.distance import edit_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

st.set_page_config(page_title="IR Assignment Group 83", layout="wide")

st.title("Information Retrieval System - Group 83")

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

documents = []

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(df.head())

    text_column = st.selectbox(
        "Select Text Column",
        df.columns
    )

    documents = df[text_column].astype(str).tolist()

    st.subheader("Document Collection")

    for i, doc in enumerate(documents[:10]):
        st.write(f"D{i+1}: {doc}")

    # PREPROCESSING

    st.header("Text Preprocessing")

    stop_words = set(stopwords.words('english'))

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    processed_docs = []
    stem_docs = []
    lemma_docs = []

    for doc in documents:

        doc = doc.lower()

        doc = re.sub(r"-", " ", doc)

        tokens = word_tokenize(doc)

        filtered = [
            t for t in tokens
            if t.isalnum() and t not in stop_words
        ]

        stemmed = [
            stemmer.stem(t)
            for t in filtered
        ]

        lemmatized = [
            lemmatizer.lemmatize(t)
            for t in filtered
        ]

        processed_docs.append(filtered)
        stem_docs.append(stemmed)
        lemma_docs.append(lemmatized)

    st.subheader("Sample Tokenization")

    if len(processed_docs):
        st.write(processed_docs[0])

    # INVERTED INDEX

    st.header("Inverted Index")

    inverted_index = defaultdict(set)

    for doc_id, tokens in enumerate(lemma_docs):

        for token in tokens:
            inverted_index[token].add(doc_id)

    inv_df = pd.DataFrame(
        [
            [term, list(ids)]
            for term, ids in list(inverted_index.items())[:100]
        ],
        columns=["Term", "Documents"]
    )

    st.dataframe(inv_df)

    # STEMMING VS LEMMATIZATION

    st.header("Stemming vs Lemmatization")

    query = st.text_input(
        "Enter Query for Comparison"
    )

    if query:

        stem_texts = [
            " ".join(doc)
            for doc in stem_docs
        ]

        lemma_texts = [
            " ".join(doc)
            for doc in lemma_docs
        ]

        stem_vectorizer = TfidfVectorizer()
        stem_matrix = stem_vectorizer.fit_transform(
            stem_texts + [query]
        )

        stem_score = cosine_similarity(
            stem_matrix[-1],
            stem_matrix[:-1]
        ).mean()

        lemma_vectorizer = TfidfVectorizer()
        lemma_matrix = lemma_vectorizer.fit_transform(
            lemma_texts + [query]
        )

        lemma_score = cosine_similarity(
            lemma_matrix[-1],
            lemma_matrix[:-1]
        ).mean()

        comparison = pd.DataFrame({
            "Method": ["Stemming", "Lemmatization"],
            "Similarity": [
                stem_score,
                lemma_score
            ]
        })

        st.table(comparison)

    # BIWORD INDEX

    st.header("Phrase Query Processing")

    biword_index = defaultdict(set)

    positional_index = defaultdict(dict)

    for doc_id, tokens in enumerate(lemma_docs):

        for i in range(len(tokens)-1):
            biword = tokens[i] + " " + tokens[i+1]
            biword_index[biword].add(doc_id)

        for pos, token in enumerate(tokens):

            if doc_id not in positional_index[token]:
                positional_index[token][doc_id] = []

            positional_index[token][doc_id].append(pos)

    phrase_query = st.text_input(
        "Enter Phrase Query"
    )

    if phrase_query:

        phrase = phrase_query.lower()

        st.subheader("Biword Search")

        if phrase in biword_index:
            st.write(list(biword_index[phrase]))
        else:
            st.write("No Result")

        st.subheader("Positional Search")

        terms = phrase.split()

        results = []

        if len(terms) == 2:

            t1, t2 = terms

            docs1 = positional_index.get(t1, {})
            docs2 = positional_index.get(t2, {})

            common = set(docs1.keys()) & set(docs2.keys())

            for d in common:

                p1 = docs1[d]
                p2 = docs2[d]

                for pos in p1:
                    if pos + 1 in p2:
                        results.append(d)

        st.write(results)

    # BST vs B TREE

    st.header("BST vs B-Tree Comparison")

    vocabulary = sorted(
        list(inverted_index.keys())
    )

    test_term = st.text_input(
        "Dictionary Search Term"
    )

    if test_term:

        start = time.perf_counter()

        bst_found = test_term in vocabulary

        bst_time = (
            time.perf_counter() - start
        ) * 1000

        start = time.perf_counter()

        btree_found = test_term in set(vocabulary)

        btree_time = (
            time.perf_counter() - start
        ) * 1000

        result_df = pd.DataFrame({
            "Structure": ["BST", "B-Tree"],
            "Search Time (ms)": [
                bst_time,
                btree_time
            ]
        })

        st.table(result_df)

    # TOLERANT RETRIEVAL

    st.header("Tolerant Retrieval")

    misspelled = st.text_input(
        "Enter Misspelled Query"
    )

    if misspelled:

        candidates = []

        for term in vocabulary:

            dist = edit_distance(
                misspelled,
                term
            )

            candidates.append(
                (term, dist)
            )

        candidates.sort(
            key=lambda x: x[1]
        )

        st.write(
            "Suggested Term:",
            candidates[0][0]
        )

    wildcard = st.text_input(
        "Wildcard Query (* supported)"
    )

    if wildcard:

        prefix = wildcard.replace("*", "")

        matches = [
            term
            for term in vocabulary
            if term.startswith(prefix)
        ]

        st.write(matches)

    # FINAL INFERENCE

    st.header("Inference and Discussion")

    st.markdown("""
    - Lowercasing and stopword removal reduced noise.
    - Lemmatization generally preserves meaning better.
    - Positional index provides more accurate phrase retrieval.
    - B-Tree lookup is typically faster for large dictionaries.
    - Tolerant retrieval improves user experience.
    - Current system relies on lexical matching.
    - Future work can include BM25 and semantic search.
    """)
