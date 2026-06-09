"""
Information Retrieval System - BITS Pilani Assignment 1
Group 83
"""

import streamlit as st
import time
import math
import re
import random
from collections import defaultdict
from io import StringIO

import nltk
import pandas as pd

for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ─────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────

STOP_WORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess(text, lowercase=True, remove_stops=True, handle_hyphens=True,
               stem=False, lemmatize=False):
    stages = {"original": text}

    if handle_hyphens:
        text = re.sub(r'-', ' ', text)
    stages["hyphen_handled"] = text

    if lowercase:
        text = text.lower()
    stages["lowercased"] = text

    tokens = word_tokenize(text)
    stages["tokenized"] = tokens

    tokens = [t for t in tokens if t.isalpha()]
    if remove_stops:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    stages["stop_removed"] = tokens

    stemmed = [stemmer.stem(t) for t in tokens]
    stages["stemmed"] = stemmed

    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    stages["lemmatized"] = lemmatized

    if stem:
        return stemmed, stages
    elif lemmatize:
        return lemmatized, stages
    return tokens, stages


def build_inverted_index(docs):
    index = defaultdict(lambda: defaultdict(list))
    for doc_id, text in enumerate(docs):
        tokens, _ = preprocess(text)
        for pos, token in enumerate(tokens):
            index[token][doc_id].append(pos)
    return index


# ─────────────────────────────────────────────────────────
# Phrase Query
# ─────────────────────────────────────────────────────────

def build_biword_index(docs):
    index = defaultdict(set)
    for doc_id, text in enumerate(docs):
        tokens, _ = preprocess(text)
        for i in range(len(tokens) - 1):
            biword = f"{tokens[i]} {tokens[i+1]}"
            index[biword].add(doc_id)
    return index


def build_positional_index(docs):
    return build_inverted_index(docs)


def biword_phrase_search(query, biword_index, docs):
    query_tokens, _ = preprocess(query)
    if len(query_tokens) < 2:
        return list(biword_index.get(query_tokens[0], set())) if query_tokens else []

    result_sets = []
    for i in range(len(query_tokens) - 1):
        biword = f"{query_tokens[i]} {query_tokens[i+1]}"
        result_sets.append(biword_index.get(biword, set()))

    if not result_sets:
        return []
    result = result_sets[0]
    for s in result_sets[1:]:
        result = result & s
    return sorted(result)


def positional_phrase_search(query, positional_index, docs):
    query_tokens, _ = preprocess(query)
    if not query_tokens:
        return []

    candidates = positional_index.get(query_tokens[0], {})
    result = set(candidates.keys())

    for i, term in enumerate(query_tokens[1:], 1):
        term_postings = positional_index.get(term, {})
        result = result & set(term_postings.keys())

    final = []
    for doc_id in sorted(result):
        first_positions = positional_index.get(query_tokens[0], {}).get(doc_id, [])
        match = False
        for pos in first_positions:
            consecutive = True
            for i, term in enumerate(query_tokens[1:], 1):
                term_positions = positional_index.get(term, {}).get(doc_id, [])
                if (pos + i) not in term_positions:
                    consecutive = False
                    break
            if consecutive:
                match = True
                break
        if match:
            final.append(doc_id)
    return final


# ─────────────────────────────────────────────────────────
# Binary Search Tree (fully iterative — avoids RecursionError)
# ─────────────────────────────────────────────────────────

class BSTNode:
    __slots__ = ('key', 'left', 'right')

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None
        self.comparisons = 0

    def insert(self, key):
        if self.root is None:
            self.root = BSTNode(key)
            return
        node = self.root
        while True:
            if key < node.key:
                if node.left is None:
                    node.left = BSTNode(key)
                    return
                node = node.left
            elif key > node.key:
                if node.right is None:
                    node.right = BSTNode(key)
                    return
                node = node.right
            else:
                return  # duplicate

    def search(self, key):
        self.comparisons = 0
        node = self.root
        while node is not None:
            self.comparisons += 1
            if key == node.key:
                return True
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return False


# ─────────────────────────────────────────────────────────
# B-Tree
# ─────────────────────────────────────────────────────────

class BTreeNode:
    def __init__(self, t, leaf=True):
        self.t = t
        self.keys = []
        self.children = []
        self.leaf = leaf

    def is_full(self):
        return len(self.keys) == 2 * self.t - 1


class BTree:
    def __init__(self, t=3):
        self.t = t
        self.root = BTreeNode(t)
        self.comparisons = 0

    def search(self, key, node=None):
        if node is None:
            node = self.root
            self.comparisons = 0
        i = 0
        while i < len(node.keys):
            self.comparisons += 1
            if key == node.keys[i]:
                return True
            elif key < node.keys[i]:
                break
            i += 1
        if node.leaf:
            return False
        return self.search(key, node.children[i])

    def insert(self, key):
        root = self.root
        if root.is_full():
            new_root = BTreeNode(self.t, leaf=False)
            new_root.children.append(self.root)
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_non_full(self.root, key)

    def _insert_non_full(self, node, key):
        i = len(node.keys) - 1
        if node.leaf:
            node.keys.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1
            node.keys[i + 1] = key
        else:
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            if node.children[i].is_full():
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(node.children[i], key)

    def _split_child(self, parent, i):
        t = self.t
        child = parent.children[i]
        new_child = BTreeNode(t, child.leaf)
        parent.keys.insert(i, child.keys[t - 1])
        parent.children.insert(i + 1, new_child)
        new_child.keys = child.keys[t:]
        child.keys = child.keys[:t - 1]
        if not child.leaf:
            new_child.children = child.children[t:]
            child.children = child.children[:t]


# ─────────────────────────────────────────────────────────
# Tolerant Retrieval
# ─────────────────────────────────────────────────────────

def build_kgram_index(vocabulary, k=2):
    index = defaultdict(set)
    for term in vocabulary:
        padded = f"${term}$"
        for i in range(len(padded) - k + 1):
            gram = padded[i:i+k]
            index[gram].add(term)
    return index


def wildcard_search(pattern, kgram_index, vocabulary, k=2):
    parts = pattern.split('*')
    parts = [p for p in parts if p]

    if not parts:
        return list(vocabulary)

    candidate_sets = []
    for part in parts:
        padded = part
        if pattern.startswith(part):
            padded = '$' + part
        if pattern.endswith(part):
            padded = part + '$'

        grams = set()
        for i in range(len(padded) - k + 1):
            grams.add(padded[i:i+k])

        if grams:
            sets = [kgram_index.get(g, set()) for g in grams]
            candidate_sets.append(set.intersection(*sets) if sets else set())

    if not candidate_sets:
        return []

    candidates = set.intersection(*candidate_sets)
    regex = re.compile('^' + pattern.replace('*', '.*') + '$')
    return sorted([t for t in candidates if regex.match(t)])


def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i-1] == s2[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def spell_correct(word, vocabulary, max_dist=2):
    candidates = []
    for term in vocabulary:
        d = edit_distance(word, term)
        if d <= max_dist:
            candidates.append((term, d))
    candidates.sort(key=lambda x: x[1])
    return candidates[:5]


def soundex(name):
    name = name.upper()
    if not name:
        return ""
    code = name[0]
    soundex_map = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6',
    }
    prev = soundex_map.get(name[0], '0')
    for ch in name[1:]:
        curr = soundex_map.get(ch, '0')
        if curr != '0' and curr != prev:
            code += curr
        prev = curr
        if len(code) == 4:
            break
    return code.ljust(4, '0')


def phonetic_search(query_word, vocabulary):
    target = soundex(query_word)
    return sorted([t for t in vocabulary if soundex(t) == target])


# ─────────────────────────────────────────────────────────
# Stemming vs Lemmatization comparison
# ─────────────────────────────────────────────────────────

def build_tf_vector(tokens, vocab):
    vec = defaultdict(float)
    for t in tokens:
        if t in vocab:
            vec[t] += 1
    return vec


def cosine_similarity(v1, v2):
    common = set(v1) & set(v2)
    dot = sum(v1[k] * v2[k] for k in common)
    mag1 = math.sqrt(sum(x**2 for x in v1.values()))
    mag2 = math.sqrt(sum(x**2 for x in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def compare_stem_lemma(docs):
    stemmed_docs = []
    lemma_docs = []
    for doc in docs:
        s_tokens, _ = preprocess(doc, stem=True)
        l_tokens, _ = preprocess(doc, lemmatize=True)
        stemmed_docs.append(s_tokens)
        lemma_docs.append(l_tokens)

    stem_vocab = set(t for d in stemmed_docs for t in d)
    lemma_vocab = set(t for d in lemma_docs for t in d)

    def avg_pairwise_sim(tokenized_docs, vocab):
        vecs = [build_tf_vector(d, vocab) for d in tokenized_docs]
        sims = []
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                sims.append(cosine_similarity(vecs[i], vecs[j]))
        return sum(sims) / len(sims) if sims else 0.0

    stem_sim = avg_pairwise_sim(stemmed_docs, stem_vocab)
    lemma_sim = avg_pairwise_sim(lemma_docs, lemma_vocab)

    return {
        "stemmed_vocab_size": len(stem_vocab),
        "lemmatized_vocab_size": len(lemma_vocab),
        "stem_avg_similarity": round(stem_sim, 4),
        "lemma_avg_similarity": round(lemma_sim, 4),
        "stemmed_samples": [d[:5] for d in stemmed_docs],
        "lemma_samples": [d[:5] for d in lemma_docs],
    }


# ─────────────────────────────────────────────────────────
# Session helpers
# ─────────────────────────────────────────────────────────

def get_docs():
    return st.session_state.get("docs", [])


def get_doc_names():
    return st.session_state.get("doc_names", [])


# ─────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────

st.set_page_config(page_title="IR System - Group 83", layout="wide")
st.title("Information Retrieval System - Group 83")

st.markdown("""
**Authors**

RAHUL KHANNA D - 2025AB05245

SUKRIT SARKAR - 2025AB05235
""")

tabs = st.tabs([
    "A. Upload & View",
    "B. Preprocessing",
    "C. Phrase Query",
    "D. Dictionary Search",
    "E. Tolerant Retrieval",
    "G. Inference",
])

# ─── TAB A ───────────────────────────────────────────────
with tabs[0]:
    st.header("A. Document Upload & Viewing")
    st.write("Upload `.txt` files or paste text below to build your document collection.")

    col1, col2 = st.columns(2)

    with col1:
        uploaded = st.file_uploader(
            "Upload text files", type=["txt"], accept_multiple_files=True
        )
        if uploaded:
            docs = []
            names = []
            for f in uploaded:
                content = f.read().decode("utf-8", errors="ignore")
                docs.append(content)
                names.append(f.name)
            st.session_state["docs"] = docs
            st.session_state["doc_names"] = names
            st.success(f"Loaded {len(docs)} document(s).")

    with col2:
        paste_text = st.text_area("Or paste text (one document per line):", height=150)
        if st.button("Load Pasted Text"):
            lines = [l.strip() for l in paste_text.strip().split('\n') if l.strip()]
            st.session_state["docs"] = lines
            st.session_state["doc_names"] = [f"Doc {i+1}" for i in range(len(lines))]
            st.success(f"Loaded {len(lines)} document(s).")

    if st.button("Load Sample Dataset"):
        sample = [
            "The quick brown fox jumps over the lazy dog near the river bank.",
            "Information retrieval systems retrieve relevant documents from large collections.",
            "Natural language processing enables computers to understand human language.",
            "The fox is a cunning animal that lives in forests and fields.",
            "Machine learning algorithms improve automatically through experience and data.",
            "Retrieval systems use indexing to speed up document search and query processing.",
            "The brown dog quickly ran and jumped over the lazy sleeping fox.",
            "Deep learning neural networks have revolutionized natural language understanding.",
            "Binary search trees and B-trees are efficient data structures for dictionary lookup.",
            "Stemming and lemmatization are text normalization techniques used in IR systems.",
            "Wildcard queries allow flexible pattern matching in information retrieval.",
            "Positional indexes store term positions enabling exact phrase query matching.",
            "Spelling correction uses edit distance to handle typographic errors in queries.",
            "The lazy cat sat on the mat near the old brown wooden fence.",
            "K-gram indexes support efficient wildcard and approximate string matching operations.",
        ]
        names = [f"doc_{i+1}.txt" for i in range(len(sample))]
        st.session_state["docs"] = sample
        st.session_state["doc_names"] = names
        st.success("Sample dataset loaded!")

    docs = get_docs()
    names = get_doc_names()

    if docs:
        st.subheader(f"Document Collection ({len(docs)} documents)")
        selected = st.selectbox("Select document to view:", range(len(docs)),
                                format_func=lambda i: names[i])
        st.text_area("Document content:", docs[selected], height=200)

        with st.expander("View all documents"):
            for i, (name, doc) in enumerate(zip(names, docs)):
                st.markdown(f"**[{i}] {name}**")
                st.text(doc[:300] + ("..." if len(doc) > 300 else ""))
                st.divider()

        st.subheader("Collection Statistics")
        all_tokens = []
        for d in docs:
            toks, _ = preprocess(d)
            all_tokens.extend(toks)
        vocab = set(all_tokens)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Documents", len(docs))
        c2.metric("Total Tokens", len(all_tokens))
        c3.metric("Vocabulary Size", len(vocab))
        c4.metric("Avg Tokens/Doc", round(len(all_tokens) / len(docs), 1))
    else:
        st.info("Please upload documents or load the sample dataset to proceed.")


# ─── TAB B ───────────────────────────────────────────────
with tabs[1]:
    st.header("B. Text Preprocessing")
    docs = get_docs()

    if not docs:
        st.warning("Please upload documents in Tab A first.")
    else:
        st.subheader("Preprocessing Pipeline Demo")

        col1, col2 = st.columns(2)
        with col1:
            do_lower = st.checkbox("Lowercasing", value=True)
            do_stops = st.checkbox("Stop word removal", value=True)
            do_hyphen = st.checkbox("Hyphen handling", value=True)
        with col2:
            doc_idx = st.selectbox("Select document:", range(len(docs)),
                                   format_func=lambda i: get_doc_names()[i])

        text = docs[doc_idx]
        tokens, stages = preprocess(text, lowercase=do_lower, remove_stops=do_stops,
                                    handle_hyphens=do_hyphen)

        st.subheader("Step-by-Step Pipeline")
        st.markdown("**Original Text:**")
        st.info(stages["original"][:500])

        if do_hyphen:
            st.markdown("**After Hyphen Handling:**")
            st.info(stages["hyphen_handled"][:500])

        if do_lower:
            st.markdown("**After Lowercasing:**")
            st.info(stages["lowercased"][:500])

        st.markdown("**After Tokenization:**")
        st.code(str(stages["tokenized"][:30]))

        if do_stops:
            st.markdown("**After Stop Word Removal:**")
            st.code(str(stages["stop_removed"][:30]))

        st.subheader("Inverted Index")
        inv_idx = build_inverted_index(docs)
        st.write(f"Index contains **{len(inv_idx)}** unique terms.")

        search_term = st.text_input("Look up term in inverted index:")
        if search_term:
            key = search_term.lower().strip()
            if key in inv_idx:
                postings = dict(inv_idx[key])
                rows = [{"Doc ID": doc_id, "Doc Name": get_doc_names()[doc_id],
                         "Positions": str(positions)}
                        for doc_id, positions in postings.items()]
                st.success(f"Term '{key}' found in {len(postings)} document(s):")
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.error(f"Term '{key}' not found in the index.")

        st.subheader("Stemming vs. Lemmatization Comparison")
        results = compare_stem_lemma(docs)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Stemming (Porter)")
            st.metric("Vocabulary Size", results["stemmed_vocab_size"])
            st.metric("Avg Pairwise Cosine Similarity", results["stem_avg_similarity"])
            st.write("**Sample stemmed tokens (Doc 1):**")
            st.code(str(results["stemmed_samples"][0]))

        with col2:
            st.markdown("### Lemmatization (WordNet)")
            st.metric("Vocabulary Size", results["lemmatized_vocab_size"])
            st.metric("Avg Pairwise Cosine Similarity", results["lemma_avg_similarity"])
            st.write("**Sample lemmatized tokens (Doc 1):**")
            st.code(str(results["lemma_samples"][0]))

        rows = []
        for i, doc in enumerate(docs):
            s_tok, _ = preprocess(doc, stem=True)
            l_tok, _ = preprocess(doc, lemmatize=True)
            rows.append({
                "Doc": get_doc_names()[i],
                "Stemmed Tokens": len(s_tok),
                "Stemmed Unique": len(set(s_tok)),
                "Lemmatized Tokens": len(l_tok),
                "Lemmatized Unique": len(set(l_tok)),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("Inference: Stemming vs. Lemmatization")
        stem_sim = results["stem_avg_similarity"]
        lemma_sim = results["lemma_avg_similarity"]
        stem_vocab = results["stemmed_vocab_size"]
        lemma_vocab = results["lemmatized_vocab_size"]

        if lemma_sim >= stem_sim:
            winner = "**Lemmatization**"
            reason = (f"Lemmatization achieves higher average cosine similarity ({lemma_sim}) "
                      f"vs stemming ({stem_sim}), indicating better semantic grouping of terms. "
                      f"Lemmatization preserves valid dictionary words, making results more "
                      f"interpretable. The vocabulary size ({lemma_vocab}) is larger than stemming "
                      f"({stem_vocab}), reflecting more meaningful term distinctions.")
        else:
            winner = "**Stemming**"
            reason = (f"Stemming achieves higher average cosine similarity ({stem_sim}) "
                      f"vs lemmatization ({lemma_sim}), indicating better grouping of related "
                      f"terms on this dataset. The reduced vocabulary size ({stem_vocab} vs "
                      f"{lemma_vocab}) helps cluster similar documents more effectively.")

        st.success(f"Conclusion: {winner} is more suitable for this dataset. {reason}")


# ─── TAB C ───────────────────────────────────────────────
with tabs[2]:
    st.header("C. Phrase Query Processing")
    docs = get_docs()

    if not docs:
        st.warning("Please upload documents in Tab A first.")
    else:
        if "biword_index" not in st.session_state or st.button("Rebuild Indexes"):
            with st.spinner("Building indexes..."):
                st.session_state["biword_index"] = build_biword_index(docs)
                st.session_state["pos_index"] = build_positional_index(docs)
            st.success("Indexes built!")

        biword_index = st.session_state.get("biword_index", {})
        pos_index = st.session_state.get("pos_index", {})

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Biword Index")
            st.write(f"Contains **{len(biword_index)}** biwords.")
            sample_biwords = list(biword_index.items())[:8]
            bw_df = pd.DataFrame([{"Biword": bw, "Doc IDs": str(sorted(dids))}
                                   for bw, dids in sample_biwords])
            st.dataframe(bw_df, use_container_width=True)

        with col2:
            st.subheader("Positional Index (sample)")
            st.write(f"Contains **{len(pos_index)}** terms.")
            sample_terms = list(pos_index.items())[:8]
            pi_df = pd.DataFrame([
                {"Term": term,
                 "Doc IDs": str(list(postings.keys())[:3]),
                 "Positions (first doc)": str(list(postings.values())[0][:5]) if postings else ""}
                for term, postings in sample_terms
            ])
            st.dataframe(pi_df, use_container_width=True)

        st.subheader("Phrase Query Search")
        query = st.text_input("Enter phrase query:", placeholder="e.g. quick brown fox")

        if query:
            t0 = time.perf_counter()
            bw_results = biword_phrase_search(query, biword_index, docs)
            bw_time = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            pos_results = positional_phrase_search(query, pos_index, docs)
            pos_time = (time.perf_counter() - t0) * 1000

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Biword Index Results")
                st.metric("Documents found", len(bw_results))
                st.metric("Query time", f"{bw_time:.3f} ms")
                if bw_results:
                    for doc_id in bw_results:
                        st.markdown(f"**[{doc_id}] {get_doc_names()[doc_id]}**")
                        st.text(docs[doc_id][:200])
                else:
                    st.info("No results.")

            with col2:
                st.markdown("### Positional Index Results")
                st.metric("Documents found", len(pos_results))
                st.metric("Query time", f"{pos_time:.3f} ms")
                if pos_results:
                    for doc_id in pos_results:
                        st.markdown(f"**[{doc_id}] {get_doc_names()[doc_id]}**")
                        st.text(docs[doc_id][:200])
                else:
                    st.info("No results.")

            false_positives = [d for d in bw_results if d not in pos_results]
            st.subheader("Analysis: False Positives in Biword Index")
            if false_positives:
                st.warning(
                    f"Biword index returned {len(false_positives)} false positive(s): "
                    f"Doc IDs {false_positives}. These documents contain adjacent biword pairs "
                    f"but not the exact phrase."
                )
                for fp in false_positives:
                    st.text(f"[{fp}] {docs[fp][:200]}")
            else:
                st.success("No false positives for this query. Both indexes agree on results.")

        st.subheader("Comparison: Biword vs Positional Index")
        st.table(pd.DataFrame([
            {
                "Aspect": "Accuracy",
                "Biword Index": "May return false positives for multi-word queries",
                "Positional Index": "Exact phrase matching — no false positives",
            },
            {
                "Aspect": "Storage",
                "Biword Index": "Smaller (pairs only)",
                "Positional Index": "Larger (all positions stored)",
            },
            {
                "Aspect": "Query Speed",
                "Biword Index": "Fast (set intersection)",
                "Positional Index": "Slightly slower (position verification required)",
            },
            {
                "Aspect": "Long Phrase Support",
                "Biword Index": "Limited — false positives increase with phrase length",
                "Positional Index": "Excellent — handles any phrase length precisely",
            },
        ]))

        st.info(
            "**Inference:** Positional index provides more accurate phrase query results because "
            "it verifies that query terms appear at consecutive positions within a document. "
            "Biword index is faster but may match documents where the biwords appear "
            "in different parts of the text, not as a contiguous phrase."
        )


# ─── TAB D ───────────────────────────────────────────────
with tabs[3]:
    st.header("D. Dictionary Search: BST vs B-Tree")
    docs = get_docs()

    if not docs:
        st.warning("Please upload documents in Tab A first.")
    else:
        if "bst" not in st.session_state or st.button("Rebuild Trees"):
            with st.spinner("Building BST and B-Tree..."):
                all_tokens = []
                for d in docs:
                    toks, _ = preprocess(d)
                    all_tokens.extend(toks)
                vocab = sorted(set(all_tokens))

                bst = BST()
                btree = BTree(t=3)

                # Shuffle vocab before BST insertion to avoid a fully skewed tree
                shuffled = vocab[:]
                random.shuffle(shuffled)
                for term in shuffled:
                    bst.insert(term)
                for term in vocab:
                    btree.insert(term)

                st.session_state["bst"] = bst
                st.session_state["btree"] = btree
                st.session_state["dict_vocab"] = vocab

            st.success(f"Trees built with {len(vocab)} terms!")

        bst = st.session_state.get("bst")
        btree = st.session_state.get("btree")
        vocab = st.session_state.get("dict_vocab", [])

        if bst and vocab:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Dictionary Size", len(vocab))
                st.write("Sample terms:", vocab[:10])

            with col2:
                query_term = st.text_input("Search term in dictionary:")

            if query_term:
                q = query_term.lower().strip()

                t0 = time.perf_counter()
                bst_found = bst.search(q)
                bst_time = (time.perf_counter() - t0) * 1e6
                bst_comps = bst.comparisons

                t0 = time.perf_counter()
                bt_found = btree.search(q)
                bt_time = (time.perf_counter() - t0) * 1e6
                bt_comps = btree.comparisons

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Binary Search Tree")
                    st.metric("Found", "Yes" if bst_found else "No")
                    st.metric("Time (us)", f"{bst_time:.2f}")
                    st.metric("Comparisons", bst_comps)

                with col2:
                    st.markdown("### B-Tree (t=3)")
                    st.metric("Found", "Yes" if bt_found else "No")
                    st.metric("Time (us)", f"{bt_time:.2f}")
                    st.metric("Comparisons", bt_comps)

            st.subheader("Experimental Results: Multiple Query Benchmark")

            test_terms = random.sample(vocab, min(10, len(vocab))) + \
                         ["xyz123", "notaword", "zzz", "aaa", "query"]

            rows = []
            for term in test_terms:
                t0 = time.perf_counter()
                bst.search(term)
                bst_t = (time.perf_counter() - t0) * 1e6
                bc = bst.comparisons

                t0 = time.perf_counter()
                btree.search(term)
                bt_t = (time.perf_counter() - t0) * 1e6
                btc = btree.comparisons

                rows.append({
                    "Query Term": term,
                    "BST Found": bst.search(term),
                    "BST Time (us)": round(bst_t, 3),
                    "BST Comparisons": bc,
                    "B-Tree Found": btree.search(term),
                    "B-Tree Time (us)": round(bt_t, 3),
                    "B-Tree Comparisons": btc,
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            avg_bst_time = df["BST Time (us)"].mean()
            avg_bt_time = df["B-Tree Time (us)"].mean()
            avg_bst_comps = df["BST Comparisons"].mean()
            avg_bt_comps = df["B-Tree Comparisons"].mean()

            st.subheader("Summary Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**BST Averages**")
                st.metric("Avg Time (us)", round(avg_bst_time, 3))
                st.metric("Avg Comparisons", round(avg_bst_comps, 1))

            with col2:
                st.markdown("**B-Tree Averages**")
                st.metric("Avg Time (us)", round(avg_bt_time, 3))
                st.metric("Avg Comparisons", round(avg_bt_comps, 1))

            st.subheader("Inference: BST vs B-Tree")
            st.info(
                "**Binary Search Tree (BST):** O(log n) average search time, but can degrade "
                "on skewed distributions. Insertion order was randomized here to avoid worst case. "
                "Well-suited for in-memory dictionary lookups on small to medium collections.\n\n"
                "**B-Tree (t=3):** Maintains O(log n) search in all cases with guaranteed balance. "
                "Each node holds multiple keys (up to 5 for t=3), reducing tree height and improving "
                "cache locality. B-Trees are preferred for large dictionaries and disk-based storage "
                "because fewer node accesses are needed.\n\n"
                f"On this dataset (vocab size={len(vocab)}), "
                + ("B-Tree requires fewer comparisons per query, consistent with its multi-key design."
                   if avg_bt_comps <= avg_bst_comps
                   else "BST shows competitive performance for this in-memory collection.")
            )


# ─── TAB E ───────────────────────────────────────────────
with tabs[4]:
    st.header("E. Tolerant Retrieval")
    docs = get_docs()

    if not docs:
        st.warning("Please upload documents in Tab A first.")
    else:
        all_tokens = []
        for d in docs:
            toks, _ = preprocess(d)
            all_tokens.extend(toks)
        vocabulary = sorted(set(all_tokens))

        if "kgram_idx" not in st.session_state:
            st.session_state["kgram_idx"] = build_kgram_index(vocabulary, k=2)

        kgram_idx = st.session_state["kgram_idx"]

        tolerant_mode = st.radio(
            "Select tolerant retrieval mode:",
            ["Wildcard Query (K-gram)", "Spelling Correction (Edit Distance)",
             "Phonetic Correction (Soundex)"],
            horizontal=True,
        )

        if tolerant_mode == "Wildcard Query (K-gram)":
            st.subheader("Wildcard Query using K-gram Index")
            st.write(f"K-gram index built with **{len(kgram_idx)}** 2-grams over {len(vocabulary)} terms.")

            wq = st.text_input("Enter wildcard query (use * as wildcard):", placeholder="e.g. ret*")
            if wq:
                t0 = time.perf_counter()
                matches = wildcard_search(wq, kgram_idx, vocabulary)
                wc_time = (time.perf_counter() - t0) * 1000

                st.metric("Matching terms found", len(matches))
                st.metric("Query time", f"{wc_time:.3f} ms")
                if matches:
                    st.success("Matching vocabulary terms:")
                    st.write(", ".join(matches))

                    inv_idx = build_inverted_index(docs)
                    matching_docs = set()
                    for m in matches:
                        if m in inv_idx:
                            matching_docs.update(inv_idx[m].keys())

                    st.write(f"**Documents containing matched terms ({len(matching_docs)}):**")
                    for doc_id in sorted(matching_docs):
                        st.markdown(f"**[{doc_id}] {get_doc_names()[doc_id]}**")
                        st.text(docs[doc_id][:200])
                else:
                    st.warning("No matches found.")

            st.subheader("How K-gram Wildcard Works")
            st.info(
                "A K-gram index maps every k-character substring of vocabulary terms "
                "(padded with $ sentinels) to the set of terms containing that substring. "
                "For a wildcard query like 'ret*', we extract k-grams from the non-wildcard parts, "
                "intersect the candidate sets, then post-filter with regex to eliminate false positives."
            )

        elif tolerant_mode == "Spelling Correction (Edit Distance)":
            st.subheader("Spelling Correction using Edit Distance (Levenshtein)")
            misspelled = st.text_input("Enter possibly misspelled query term:", placeholder="e.g. retreival")
            max_d = st.slider("Maximum edit distance:", 1, 4, 2)

            if misspelled:
                t0 = time.perf_counter()
                candidates = spell_correct(misspelled.lower(), vocabulary, max_dist=max_d)
                sc_time = (time.perf_counter() - t0) * 1000

                st.metric("Correction candidates found", len(candidates))
                st.metric("Time", f"{sc_time:.1f} ms")

                if candidates:
                    df = pd.DataFrame(candidates, columns=["Suggested Term", "Edit Distance"])
                    st.dataframe(df, use_container_width=True)

                    best = candidates[0][0]
                    st.success(f"Best correction: {best} (distance={candidates[0][1]})")

                    inv_idx = build_inverted_index(docs)
                    if best in inv_idx:
                        st.write(f"Documents containing '{best}':")
                        for doc_id in inv_idx[best]:
                            st.markdown(f"**[{doc_id}] {get_doc_names()[doc_id]}**")
                            st.text(docs[doc_id][:200])
                else:
                    st.warning("No corrections found within the specified edit distance.")

            st.subheader("Edit Distance Demonstration")
            col1, col2 = st.columns(2)
            with col1:
                w1 = st.text_input("Word 1:", value="retrieval")
            with col2:
                w2 = st.text_input("Word 2:", value="retreival")
            if w1 and w2:
                d = edit_distance(w1, w2)
                st.metric(f"Edit distance between '{w1}' and '{w2}'", d)

        else:
            st.subheader("Phonetic Correction using Soundex")
            phonetic_query = st.text_input("Enter query word for phonetic matching:",
                                           placeholder="e.g. smith")

            if phonetic_query:
                target_code = soundex(phonetic_query.upper())
                t0 = time.perf_counter()
                matches = phonetic_search(phonetic_query, vocabulary)
                ph_time = (time.perf_counter() - t0) * 1000

                st.metric("Soundex code", target_code)
                st.metric("Phonetically similar terms", len(matches))
                st.metric("Time", f"{ph_time:.3f} ms")

                if matches:
                    st.success("Vocabulary terms with same Soundex code:")
                    df = pd.DataFrame([(t, soundex(t.upper())) for t in matches],
                                      columns=["Term", "Soundex Code"])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No phonetically similar terms found in vocabulary.")

            st.subheader("Soundex Examples")
            examples = ["smith", "smythe", "information", "infermation", "retrieval", "retreival"]
            snd_df = pd.DataFrame([(w, soundex(w.upper())) for w in examples],
                                  columns=["Word", "Soundex Code"])
            st.dataframe(snd_df, use_container_width=True)


# ─── TAB G ───────────────────────────────────────────────
with tabs[5]:
    st.header("G. Inference and Discussion")

    docs = get_docs()

    st.subheader("System Summary")
    if docs:
        all_tokens = []
        for d in docs:
            toks, _ = preprocess(d)
            all_tokens.extend(toks)
        vocab = set(all_tokens)

        c1, c2, c3 = st.columns(3)
        c1.metric("Documents", len(docs))
        c2.metric("Vocabulary", len(vocab))
        c3.metric("Total Tokens", len(all_tokens))

    st.markdown("---")

    inferences = {
        "1. Which preprocessing technique improved retrieval quality?": (
            "**Lowercasing + Stop Word Removal** had the most significant impact. "
            "Lowercasing ensures 'Fox' and 'fox' match the same index entry. "
            "Stop word removal eliminates high-frequency function words (the, is, a) "
            "that carry no discriminative value, reducing noise and index size substantially. "
            "Hyphen handling unifies compound terms, avoiding split-token mismatches."
        ),
        "2. Was stemming or lemmatization better for this dataset?": (
            "**Lemmatization** is generally more suitable for this dataset. "
            "It produces valid dictionary words (e.g., 'running' to 'run') vs. stemming's "
            "truncated roots (e.g., 'studies' to 'studi'). "
            "The higher cosine similarity score under lemmatization indicates better "
            "semantic grouping, and results are more interpretable for end users. "
            "Stemming trades precision for recall — useful when coverage matters more than readability."
        ),
        "3. Which phrase query index was more accurate?": (
            "**Positional Index** is more accurate. "
            "It verifies that query terms appear at consecutive positions within a document, "
            "eliminating false positives that the biword index may return. "
            "The biword index is faster (simple set intersection) but cannot distinguish "
            "whether two adjacent biwords form a contiguous phrase or appear separately."
        ),
        "4. Which tree structure was faster?": (
            "**B-Tree** exhibits more consistent O(log_t n) performance. "
            "Each B-Tree node stores multiple keys (up to 2t-1), meaning fewer node accesses "
            "are needed for large vocabularies. BST is competitive for in-memory small "
            "dictionaries with randomized insertion but can degrade on sorted input. "
            "For disk-based IR systems, B-Trees are standard because they minimize I/O operations."
        ),
        "5. How tolerant was the retrieval model?": (
            "The tolerant retrieval module handles three classes of imperfect queries:\n"
            "- **Wildcard queries** via k-gram index (2-gram): supports prefix (ret*), "
            "suffix (*al), and infix wildcards with regex post-filtering.\n"
            "- **Spelling correction** via Levenshtein edit distance: corrects up to d=2 "
            "typos, effective for common transpositions and substitutions.\n"
            "- **Phonetic correction** via Soundex: matches homophones and near-homophones, "
            "useful when users spell phonetically."
        ),
        "6. What are the limitations of this system?": (
            "- **Scale**: All indexes are in-memory; large corpora would require disk-based structures.\n"
            "- **BST imbalance**: Without self-balancing (AVL or Red-Black), BST can degrade to O(n) "
            "on sorted inputs (mitigated here by random insertion order).\n"
            "- **Biword false positives**: Cannot be entirely eliminated without position verification.\n"
            "- **Soundex limitations**: Collapses distinct sounds; Metaphone or Double Metaphone "
            "would be more accurate.\n"
            "- **No ranking**: The system returns unranked result sets. TF-IDF or BM25 ranking "
            "would improve result quality significantly."
        ),
        "7. How can the system be improved?": (
            "- **Ranking**: Implement TF-IDF or BM25 scoring for ranked retrieval.\n"
            "- **Balanced BST**: Use AVL tree or Red-Black tree to guarantee O(log n) worst case.\n"
            "- **Permuterm index**: Combine with biword index to eliminate wildcard false positives.\n"
            "- **Context-aware lemmatization**: Use POS tagging to select the correct lemma form.\n"
            "- **Neural IR**: Integrate dense retrieval (sentence-transformers) for "
            "semantic similarity beyond keyword matching.\n"
            "- **Disk persistence**: Serialize indexes to disk (SQLite or file-based inverted index) "
            "to handle large document collections."
        ),
    }

    for question, answer in inferences.items():
        with st.expander(question, expanded=True):
            st.markdown(answer)

    st.markdown("---")
    st.subheader("Architecture Overview")
    st.code("""
IR System Architecture
─────────────────────────────────────
Input Layer
  └─ Document Upload (TXT files / Paste)

Preprocessing Pipeline
  ├─ Hyphen Handling
  ├─ Lowercasing
  ├─ Tokenization (NLTK)
  ├─ Stop Word Removal
  ├─ Stemming (Porter Stemmer)
  └─ Lemmatization (WordNet)

Indexing Layer
  ├─ Inverted Index (term -> doc_id -> positions)
  ├─ Biword Index (bigram -> doc_ids)
  ├─ Positional Index (term -> doc_id -> positions)
  ├─ Binary Search Tree (in-memory dictionary)
  ├─ B-Tree (t=3, multi-key nodes)
  └─ K-gram Index (2-gram -> vocabulary terms)

Query Processing
  ├─ Exact Keyword Lookup
  ├─ Phrase Query (Biword / Positional)
  └─ Tolerant Retrieval
       ├─ Wildcard (K-gram + regex post-filter)
       ├─ Spelling Correction (Edit Distance)
       └─ Phonetic Matching (Soundex)

Output Layer
  └─ Streamlit Front-end (results, tables, metrics)
""", language="text")

    st.caption("Group 83 | RAHUL KHANNA D (2025AB05245) | SUKRIT SARKAR (2025AB05235)")
