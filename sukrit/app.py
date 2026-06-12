"""
Information Retrieval System - BITS Pilani Assignment 2
Group 83
"""

import streamlit as st
import time
import math
import re
import random
from collections import defaultdict
from pathlib import Path

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
# Constants
# ─────────────────────────────────────────────────────────

APP_DIR = Path(__file__).resolve().parent
DATASET_DIR = APP_DIR / "dataset"

STOP_WORDS = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

FIXED_BENCHMARK_QUERIES = [
    "information", "retrieval", "microsoft", "election", "virus",
    "technology", "farmer", "ink", "software", "network",
    "xyz123", "notaword", "zzz",
]

PREPROCESS_TEST_QUERIES = [
    "information retrieval",
    "anti-spyware",
    "e-mail",
    "Microsoft",
    "elections",
]

STEM_LEMMA_QUERIES = [
    "running", "studies", "retrieval", "investigating", "systems",
]

# ─────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────

def default_preprocess_config():
    return {
        "lowercase": True,
        "remove_stops": True,
        "handle_hyphens": True,
        "normalization": "none",  # none | stem | lemma
    }


def preprocess(text, lowercase=True, remove_stops=True, handle_hyphens=True,
               stem=False, lemmatize=False, normalization=None):
    if normalization == "stem":
        stem, lemmatize = True, False
    elif normalization == "lemma":
        stem, lemmatize = False, True
    elif normalization == "none":
        stem, lemmatize = False, False

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
    if lemmatize:
        return lemmatized, stages
    return tokens, stages


def preprocess_with_config(text, config):
    norm = config.get("normalization", "none")
    return preprocess(
        text,
        lowercase=config.get("lowercase", True),
        remove_stops=config.get("remove_stops", True),
        handle_hyphens=config.get("handle_hyphens", True),
        normalization=norm,
    )


def build_inverted_index(docs, config=None):
    config = config or default_preprocess_config()
    index = defaultdict(lambda: defaultdict(list))
    for doc_id, text in enumerate(docs):
        tokens, _ = preprocess_with_config(text, config)
        for pos, token in enumerate(tokens):
            index[token][doc_id].append(pos)
    return index


def boolean_and_search(query, inverted_index, config=None):
    query_tokens, _ = preprocess_with_config(query, config or default_preprocess_config())
    if not query_tokens:
        return []

    result = None
    for term in query_tokens:
        postings = set(inverted_index.get(term, {}).keys())
        result = postings if result is None else result & postings

    return sorted(result or [])


# ─────────────────────────────────────────────────────────
# Phrase Query
# ─────────────────────────────────────────────────────────

def build_biword_index(docs, config=None):
    config = config or default_preprocess_config()
    index = defaultdict(set)
    for doc_id, text in enumerate(docs):
        tokens, _ = preprocess_with_config(text, config)
        for i in range(len(tokens) - 1):
            biword = f"{tokens[i]} {tokens[i+1]}"
            index[biword].add(doc_id)
    return index


def build_positional_index(docs, config=None):
    return build_inverted_index(docs, config)


def biword_phrase_search(query, biword_index, config=None):
    query_tokens, _ = preprocess_with_config(query, config or default_preprocess_config())
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


def positional_phrase_search(query, positional_index, config=None):
    query_tokens, _ = preprocess_with_config(query, config or default_preprocess_config())
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
        for pos in first_positions:
            consecutive = True
            for i, term in enumerate(query_tokens[1:], 1):
                term_positions = positional_index.get(term, {}).get(doc_id, [])
                if (pos + i) not in term_positions:
                    consecutive = False
                    break
            if consecutive:
                final.append(doc_id)
                break
    return final


# ─────────────────────────────────────────────────────────
# Binary Search Tree
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
                return

    def search(self, key):
        self.comparisons = 0
        node = self.root
        while node is not None:
            self.comparisons += 1
            if key == node.key:
                return True
            if key < node.key:
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
            if key < node.keys[i]:
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
            gram = padded[i:i + k]
            index[gram].add(term)
    return index


def wildcard_search(pattern, kgram_index, vocabulary, k=2):
    parts = pattern.split('*')
    if not any(parts):
        return list(vocabulary)

    candidate_sets = []
    n_parts = len(parts)

    for idx, part in enumerate(parts):
        if not part:
            continue
        is_first = idx == 0
        is_last = idx == n_parts - 1
        padded = ('$' if is_first else '') + part + ('$' if is_last else '')

        grams = {padded[i:i + k] for i in range(len(padded) - k + 1)}
        if grams:
            sets = [kgram_index.get(g, set()) for g in grams]
            candidate_sets.append(set.intersection(*sets))

    if not candidate_sets:
        return []

    candidates = set.intersection(*candidate_sets)
    escaped_parts = [re.escape(p) for p in parts]
    regex = re.compile('^' + '.*'.join(escaped_parts) + '$')
    return sorted(t for t in candidates if regex.match(t))


def lookup_kgram(gram, kgram_index):
    return sorted(kgram_index.get(gram.lower(), set()))


def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
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
    return sorted(t for t in vocabulary if soundex(t) == target)


# ─────────────────────────────────────────────────────────
# Experiments & comparisons
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
    mag1 = math.sqrt(sum(x ** 2 for x in v1.values()))
    mag2 = math.sqrt(sum(x ** 2 for x in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def compare_stem_lemma_similarity(docs):
    stemmed_docs, lemma_docs = [], []
    for doc in docs:
        s_tokens, _ = preprocess(doc, stem=True)
        l_tokens, _ = preprocess(doc, lemmatize=True)
        stemmed_docs.append(s_tokens)
        lemma_docs.append(l_tokens)

    stem_vocab = {t for d in stemmed_docs for t in d}
    lemma_vocab = {t for d in lemma_docs for t in d}

    def avg_pairwise_sim(tokenized_docs, vocab):
        vecs = [build_tf_vector(d, vocab) for d in tokenized_docs]
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sims.append(cosine_similarity(vecs[i], vecs[j]))
        return sum(sims) / len(sims) if sims else 0.0

    return {
        "stemmed_vocab_size": len(stem_vocab),
        "lemmatized_vocab_size": len(lemma_vocab),
        "stem_avg_similarity": round(avg_pairwise_sim(stemmed_docs, stem_vocab), 4),
        "lemma_avg_similarity": round(avg_pairwise_sim(lemma_docs, lemma_vocab), 4),
        "stemmed_samples": [d[:5] for d in stemmed_docs],
        "lemma_samples": [d[:5] for d in lemma_docs],
    }


def compare_stem_lemma_retrieval(docs, queries):
    stem_config = {**default_preprocess_config(), "normalization": "stem"}
    lemma_config = {**default_preprocess_config(), "normalization": "lemma"}

    stem_index = build_inverted_index(docs, stem_config)
    lemma_index = build_inverted_index(docs, lemma_config)

    rows = []
    stem_hits, lemma_hits = 0, 0
    for query in queries:
        stem_results = boolean_and_search(query, stem_index, stem_config)
        lemma_results = boolean_and_search(query, lemma_index, lemma_config)
        stem_hits += len(stem_results)
        lemma_hits += len(lemma_results)
        rows.append({
            "Query": query,
            "Stem Hits": len(stem_results),
            "Lemma Hits": len(lemma_results),
            "Stem Doc IDs": str(stem_results),
            "Lemma Doc IDs": str(lemma_results),
        })

    return {
        "rows": rows,
        "total_stem_hits": stem_hits,
        "total_lemma_hits": lemma_hits,
        "winner": "stem" if stem_hits > lemma_hits else ("lemma" if lemma_hits > stem_hits else "tie"),
    }


def run_preprocessing_impact(docs, queries, base_config):
    variants = [
        ("Baseline (no preprocessing)", {"lowercase": False, "remove_stops": False,
                                       "handle_hyphens": False, "normalization": "none"}),
        ("+ Lowercasing", {**base_config, "lowercase": True, "remove_stops": False,
                           "handle_hyphens": False, "normalization": "none"}),
        ("+ Stop word removal", {**base_config, "lowercase": True, "remove_stops": True,
                                "handle_hyphens": False, "normalization": "none"}),
        ("+ Hyphen handling", {**base_config, "lowercase": True, "remove_stops": True,
                              "handle_hyphens": True, "normalization": "none"}),
        ("Full pipeline", base_config),
    ]

    rows = []
    for label, config in variants:
        index = build_inverted_index(docs, config)
        total_hits = 0
        per_query = []
        for query in queries:
            hits = boolean_and_search(query, index, config)
            total_hits += len(hits)
            per_query.append(len(hits))
        rows.append({
            "Configuration": label,
            "Total Hits": total_hits,
            "Avg Hits/Query": round(total_hits / len(queries), 2) if queries else 0,
            "Per-Query Hits": str(per_query),
        })

    best = max(rows, key=lambda r: r["Total Hits"])
    return rows, best["Configuration"]


def retrieve_from_inverted_index(term, inverted_index):
    postings = inverted_index.get(term, {})
    return sorted(postings.keys())


def run_tree_benchmark(bst, btree, vocab, inverted_index):
    rows = []
    for term in FIXED_BENCHMARK_QUERIES:
        t0 = time.perf_counter()
        bst_found = bst.search(term)
        bst_search_time = (time.perf_counter() - t0) * 1e6
        bst_comps = bst.comparisons

        t0 = time.perf_counter()
        if bst_found:
            bst_docs = retrieve_from_inverted_index(term, inverted_index)
        else:
            bst_docs = []
        bst_retrieval_time = (time.perf_counter() - t0) * 1e6

        t0 = time.perf_counter()
        bt_found = btree.search(term)
        bt_search_time = (time.perf_counter() - t0) * 1e6
        bt_comps = btree.comparisons

        t0 = time.perf_counter()
        if bt_found:
            bt_docs = retrieve_from_inverted_index(term, inverted_index)
        else:
            bt_docs = []
        bt_retrieval_time = (time.perf_counter() - t0) * 1e6

        rows.append({
            "Query Term": term,
            "BST Found": bst_found,
            "BST Search Time (us)": round(bst_search_time, 3),
            "BST Retrieval Time (us)": round(bst_retrieval_time, 3),
            "BST Comparisons": bst_comps,
            "BST Docs Retrieved": len(bst_docs),
            "B-Tree Found": bt_found,
            "B-Tree Search Time (us)": round(bt_search_time, 3),
            "B-Tree Retrieval Time (us)": round(bt_retrieval_time, 3),
            "B-Tree Comparisons": bt_comps,
            "B-Tree Docs Retrieved": len(bt_docs),
        })

    df = pd.DataFrame(rows)
    summary = {
        "avg_bst_search": df["BST Search Time (us)"].mean(),
        "avg_bt_search": df["B-Tree Search Time (us)"].mean(),
        "avg_bst_retrieval": df["BST Retrieval Time (us)"].mean(),
        "avg_bt_retrieval": df["B-Tree Retrieval Time (us)"].mean(),
        "avg_bst_comps": df["BST Comparisons"].mean(),
        "avg_bt_comps": df["B-Tree Comparisons"].mean(),
        "faster_search": "B-Tree" if df["B-Tree Search Time (us)"].mean() <= df["BST Search Time (us)"].mean() else "BST",
        "faster_retrieval": "B-Tree" if df["B-Tree Retrieval Time (us)"].mean() <= df["BST Retrieval Time (us)"].mean() else "BST",
    }
    return df, summary


# ─────────────────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────────────────

def load_project_dataset():
    if not DATASET_DIR.exists():
        return [], []
    files = sorted(DATASET_DIR.glob("*.txt"))
    docs, names = [], []
    for path in files:
        docs.append(path.read_text(encoding="utf-8", errors="ignore"))
        names.append(path.name)
    return docs, names


def set_documents(docs, names):
    st.session_state["docs"] = docs
    st.session_state["doc_names"] = names
    for key in ["biword_index", "pos_index", "bst", "btree", "dict_vocab",
                "kgram_idx", "inverted_index", "experiments"]:
        st.session_state.pop(key, None)


# ─────────────────────────────────────────────────────────
# Session helpers
# ─────────────────────────────────────────────────────────

def get_docs():
    return st.session_state.get("docs", [])


def get_doc_names():
    return st.session_state.get("doc_names", [])


def get_config_from_sidebar(prefix=""):
    key = f"{prefix}cfg"
    if key not in st.session_state:
        st.session_state[key] = default_preprocess_config()
    return st.session_state[key]


def render_preprocess_controls(key_prefix, in_sidebar=False):
    container = st.sidebar if in_sidebar else st
    with container.expander("Preprocessing Options", expanded=True):
        cfg = default_preprocess_config()
        cfg["lowercase"] = st.checkbox("Lowercasing", value=True, key=f"{key_prefix}_lower")
        cfg["remove_stops"] = st.checkbox("Stop word removal", value=True, key=f"{key_prefix}_stops")
        cfg["handle_hyphens"] = st.checkbox("Hyphen handling", value=True, key=f"{key_prefix}_hyphen")
        cfg["normalization"] = st.radio(
            "Normalization",
            ["none", "stem", "lemma"],
            format_func=lambda x: {"none": "None", "stem": "Stemming", "lemma": "Lemmatization"}[x],
            horizontal=True,
            key=f"{key_prefix}_norm",
        )
    st.session_state[f"{key_prefix}_cfg"] = cfg
    return cfg


def display_doc_results(doc_ids, docs, names, limit=200):
    if not doc_ids:
        st.info("No results.")
        return
    for doc_id in doc_ids:
        st.markdown(f"**[{doc_id}] {names[doc_id]}**")
        st.text(docs[doc_id][:limit])


def store_experiments(**kwargs):
    exp = st.session_state.get("experiments", {})
    exp.update(kwargs)
    st.session_state["experiments"] = exp


# ─────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────

st.set_page_config(page_title="IR System - Assignment 1 - Group 83", layout="wide")
st.title("Information Retrieval System — Assignment 1")
st.caption("Group 83 | RAHUL KHANNA D (2025AB05245) | SUKRIT SARKAR (2025AB05235)")

tabs = st.tabs([
    "A. Upload & View",
    "B. Preprocessing",
    "C. Query & Retrieve",
    "D. Phrase Query",
    "E. Dictionary Search",
    "F. Tolerant Retrieval",
    "G. Inference",
])

# ─── TAB A ───────────────────────────────────────────────
with tabs[0]:
    st.header("A. Document Upload & Viewing")
    st.write("Upload `.txt` files or load the project dataset from the `dataset/` folder.")

    col1, col2, col3 = st.columns(3)
    with col1:
        uploaded = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)
        if uploaded:
            docs, names = [], []
            for f in uploaded:
                docs.append(f.read().decode("utf-8", errors="ignore"))
                names.append(f.name)
            set_documents(docs, names)
            st.success(f"Loaded {len(docs)} document(s).")

    with col2:
        paste_text = st.text_area("Or paste text (one document per line):", height=120, key="paste_area")
        if st.button("Load Pasted Text"):
            lines = [l.strip() for l in paste_text.strip().split('\n') if l.strip()]
            set_documents(lines, [f"Doc {i + 1}" for i in range(len(lines))])
            st.success(f"Loaded {len(lines)} document(s).")

    with col3:
        if st.button("Load Project Dataset"):
            docs, names = load_project_dataset()
            if docs:
                set_documents(docs, names)
                st.success(f"Loaded {len(docs)} documents from `dataset/`.")
            else:
                st.error("No files found in dataset/ folder.")

    docs = get_docs()
    names = get_doc_names()

    if docs:
        st.subheader(f"Document Collection ({len(docs)} documents)")
        selected = st.selectbox("Select document to view:", range(len(docs)), format_func=lambda i: names[i])
        st.text_area("Document content:", docs[selected], height=200)

        with st.expander("View all documents"):
            for i, (name, doc) in enumerate(zip(names, docs)):
                st.markdown(f"**[{i}] {name}**")
                st.text(doc[:300] + ("..." if len(doc) > 300 else ""))
                st.divider()

        cfg = default_preprocess_config()
        all_tokens = []
        for d in docs:
            toks, _ = preprocess_with_config(d, cfg)
            all_tokens.extend(toks)
        vocab = set(all_tokens)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Documents", len(docs))
        c2.metric("Total Tokens", len(all_tokens))
        c3.metric("Vocabulary Size", len(vocab))
        c4.metric("Avg Tokens/Doc", round(len(all_tokens) / len(docs), 1))
    else:
        st.info("Please upload documents or load the project dataset to proceed.")


# ─── TAB B ───────────────────────────────────────────────
with tabs[1]:
    st.header("B. Text Preprocessing")
    docs = get_docs()

    if not docs:
        st.warning("Please upload documents in Tab A first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            do_lower = st.checkbox("Lowercasing", value=True, key="b_lower")
            do_stops = st.checkbox("Stop word removal", value=True, key="b_stops")
            do_hyphen = st.checkbox("Hyphen handling", value=True, key="b_hyphen")
        with col2:
            doc_idx = st.selectbox("Select document:", range(len(docs)), format_func=lambda i: get_doc_names()[i])

        text = docs[doc_idx]
        tokens, stages = preprocess(text, lowercase=do_lower, remove_stops=do_stops, handle_hyphens=do_hyphen)

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
        st.markdown("**After Stemming:**")
        st.code(str(stages["stemmed"][:30]))
        st.markdown("**After Lemmatization:**")
        st.code(str(stages["lemmatized"][:30]))

        st.subheader("Inverted Index")
        base_cfg = default_preprocess_config()
        inv_idx = build_inverted_index(docs, base_cfg)
        st.write(f"Index contains **{len(inv_idx)}** unique terms.")
        search_term = st.text_input("Look up term in inverted index:")
        if search_term:
            key, _ = preprocess_with_config(search_term, base_cfg)
            key = key[0] if key else search_term.lower().strip()
            if key in inv_idx:
                postings = dict(inv_idx[key])
                rows = [{"Doc ID": did, "Doc Name": get_doc_names()[did], "Positions": str(pos)}
                        for did, pos in postings.items()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.error(f"Term '{key}' not found.")

        st.subheader("Preprocessing Impact on Retrieval Quality")
        st.caption("Fixed test queries run under progressively applied preprocessing steps.")
        impact_rows, best_config = run_preprocessing_impact(docs, PREPROCESS_TEST_QUERIES, base_cfg)
        st.dataframe(pd.DataFrame(impact_rows), use_container_width=True)
        st.success(f"Best configuration by total hits: **{best_config}**")
        store_experiments(best_preprocess=best_config, preprocess_impact=impact_rows)

        st.subheader("Stemming vs. Lemmatization")
        sim_results = compare_stem_lemma_similarity(docs)
        retrieval_results = compare_stem_lemma_retrieval(docs, STEM_LEMMA_QUERIES)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Semantic Similarity (Cosine)")
            st.metric("Stem vocab size", sim_results["stemmed_vocab_size"])
            st.metric("Stem avg similarity", sim_results["stem_avg_similarity"])
            st.metric("Lemma vocab size", sim_results["lemmatized_vocab_size"])
            st.metric("Lemma avg similarity", sim_results["lemma_avg_similarity"])
        with col2:
            st.markdown("### Query Retrieval Comparison")
            st.metric("Total stem hits", retrieval_results["total_stem_hits"])
            st.metric("Total lemma hits", retrieval_results["total_lemma_hits"])
            winner_label = {"stem": "Stemming", "lemma": "Lemmatization", "tie": "Tie"}[retrieval_results["winner"]]
            st.metric("Retrieval winner", winner_label)

        st.dataframe(pd.DataFrame(retrieval_results["rows"]), use_container_width=True)

        if retrieval_results["winner"] == "lemma" or (
            retrieval_results["winner"] == "tie" and sim_results["lemma_avg_similarity"] >= sim_results["stem_avg_similarity"]
        ):
            stem_winner = "Lemmatization"
            stem_reason = (
                f"Lemmatization retrieved {retrieval_results['total_lemma_hits']} document hits vs "
                f"{retrieval_results['total_stem_hits']} for stemming across test queries. "
                f"It also achieves cosine similarity {sim_results['lemma_avg_similarity']} vs "
                f"{sim_results['stem_avg_similarity']}, producing valid dictionary words."
            )
        else:
            stem_winner = "Stemming"
            stem_reason = (
                f"Stemming retrieved {retrieval_results['total_stem_hits']} document hits vs "
                f"{retrieval_results['total_lemma_hits']} for lemmatization. "
                f"Its smaller vocabulary ({sim_results['stemmed_vocab_size']} vs "
                f"{sim_results['lemmatized_vocab_size']}) improves term conflation on this collection."
            )

        st.success(f"Conclusion: **{stem_winner}** is more suitable. {stem_reason}")
        store_experiments(
            stem_lemma_winner=stem_winner,
            stem_lemma_reason=stem_reason,
            sim_results=sim_results,
            retrieval_results=retrieval_results,
        )


# ─── TAB C ───────────────────────────────────────────────
with tabs[2]:
    st.header("C. Query & Retrieve")
    docs = get_docs()

    if not docs:
        st.warning("Please upload documents in Tab A first.")
    else:
        cfg = render_preprocess_controls("query")

        query = st.text_input("Enter search query:", placeholder="e.g. information retrieval")
        retrieval_mode = st.selectbox(
            "Retrieval technique",
            ["Boolean AND (Inverted Index)", "Phrase Query (Biword)", "Phrase Query (Positional)",
             "Wildcard Query", "Spelling Correction", "Phonetic (Soundex)"],
        )

        if st.button("Run Query", type="primary") and query:
            names = get_doc_names()
            t0 = time.perf_counter()

            if retrieval_mode == "Boolean AND (Inverted Index)":
                inv_idx = build_inverted_index(docs, cfg)
                results = boolean_and_search(query, inv_idx, cfg)
                st.session_state["last_query_detail"] = f"Terms: {preprocess_with_config(query, cfg)[0]}"

            elif retrieval_mode.startswith("Phrase Query"):
                biword_idx = build_biword_index(docs, cfg)
                pos_idx = build_positional_index(docs, cfg)
                if "Biword" in retrieval_mode:
                    results = biword_phrase_search(query, biword_idx, cfg)
                else:
                    results = positional_phrase_search(query, pos_idx, cfg)

            else:
                vocab_tokens = []
                for d in docs:
                    toks, _ = preprocess_with_config(d, cfg)
                    vocab_tokens.extend(toks)
                vocabulary = sorted(set(vocab_tokens))
                inv_idx = build_inverted_index(docs, cfg)

                if retrieval_mode == "Wildcard Query":
                    kgram_idx = build_kgram_index(vocabulary)
                    matched_terms = wildcard_search(query, kgram_idx, vocabulary)
                    results = sorted({
                        did for term in matched_terms for did in inv_idx.get(term, {})
                    })
                    st.session_state["last_query_detail"] = f"Matched terms: {matched_terms}"
                elif retrieval_mode == "Spelling Correction":
                    candidates = spell_correct(query.lower(), vocabulary)
                    if candidates:
                        best = candidates[0][0]
                        results = sorted(inv_idx.get(best, {}).keys())
                        st.session_state["last_query_detail"] = f"Corrected '{query}' → '{best}' (dist={candidates[0][1]})"
                    else:
                        results = []
                        st.session_state["last_query_detail"] = "No correction found"
                else:
                    matched_terms = phonetic_search(query, vocabulary)
                    results = sorted({
                        did for term in matched_terms for did in inv_idx.get(term, {})
                    })
                    st.session_state["last_query_detail"] = f"Phonetic matches: {matched_terms}"

            elapsed = (time.perf_counter() - t0) * 1000
            st.metric("Documents retrieved", len(results))
            st.metric("Query time", f"{elapsed:.3f} ms")
            if st.session_state.get("last_query_detail"):
                st.caption(st.session_state["last_query_detail"])
            display_doc_results(results, docs, names)


# ─── TAB D ───────────────────────────────────────────────
with tabs[3]:
    st.header("D. Phrase Query Processing")
    docs = get_docs()

    if not docs:
        st.warning("Please upload documents in Tab A first.")
    else:
        cfg = default_preprocess_config()

        if "biword_index" not in st.session_state or st.button("Rebuild Phrase Indexes"):
            with st.spinner("Building indexes..."):
                st.session_state["biword_index"] = build_biword_index(docs, cfg)
                st.session_state["pos_index"] = build_positional_index(docs, cfg)
            st.success("Indexes built.")

        biword_index = st.session_state["biword_index"]
        pos_index = st.session_state["pos_index"]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Biword Index")
            st.write(f"**{len(biword_index)}** biwords.")
            sample = list(biword_index.items())[:8]
            st.dataframe(pd.DataFrame([{"Biword": bw, "Doc IDs": str(sorted(dids))} for bw, dids in sample]),
                         use_container_width=True)
        with col2:
            st.subheader("Positional Index (sample)")
            st.write(f"**{len(pos_index)}** terms.")
            sample = list(pos_index.items())[:8]
            st.dataframe(pd.DataFrame([
                {"Term": t, "Doc IDs": str(list(p.keys())[:3]),
                 "Positions": str(list(p.values())[0][:5]) if p else ""}
                for t, p in sample
            ]), use_container_width=True)

        st.subheader("Phrase Query Search")
        query = st.text_input("Enter phrase query:", placeholder="e.g. microsoft investigating")

        if query:
            t0 = time.perf_counter()
            bw_results = biword_phrase_search(query, biword_index, cfg)
            bw_time = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            pos_results = positional_phrase_search(query, pos_index, cfg)
            pos_time = (time.perf_counter() - t0) * 1000

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Biword Results")
                st.metric("Documents found", len(bw_results))
                st.metric("Query time", f"{bw_time:.3f} ms")
                display_doc_results(bw_results, docs, get_doc_names())
            with col2:
                st.markdown("### Positional Results")
                st.metric("Documents found", len(pos_results))
                st.metric("Query time", f"{pos_time:.3f} ms")
                display_doc_results(pos_results, docs, get_doc_names())

            false_positives = [d for d in bw_results if d not in pos_results]
            st.subheader("Biword False Positives")
            if false_positives:
                st.warning(
                    f"Biword returned {len(false_positives)} false positive(s): Doc IDs {false_positives}. "
                    "Adjacent biword pairs exist but the full phrase is not contiguous."
                )
                for fp in false_positives:
                    st.text(f"[{fp}] {docs[fp][:250]}")
            else:
                st.success("No false positives for this query.")

            store_experiments(
                phrase_winner="Positional Index",
                phrase_false_positives=len(false_positives),
                last_phrase_query=query,
            )

        st.info(
            "**Inference:** Positional index verifies consecutive term positions, eliminating biword false "
            "positives where overlapping bigrams appear in different parts of a document."
        )


# ─── TAB E ───────────────────────────────────────────────
with tabs[4]:
    st.header("E. Dictionary Search: BST vs B-Tree")
    docs = get_docs()

    if not docs:
        st.warning("Please upload documents in Tab A first.")
    else:
        cfg = default_preprocess_config()

        if "bst" not in st.session_state or st.button("Rebuild Trees"):
            with st.spinner("Building BST and B-Tree..."):
                all_tokens = []
                for d in docs:
                    toks, _ = preprocess_with_config(d, cfg)
                    all_tokens.extend(toks)
                vocab = sorted(set(all_tokens))
                inv_idx = build_inverted_index(docs, cfg)

                bst, btree = BST(), BTree(t=3)
                shuffled = vocab[:]
                random.seed(42)
                random.shuffle(shuffled)
                for term in shuffled:
                    bst.insert(term)
                for term in vocab:
                    btree.insert(term)

                st.session_state.update({
                    "bst": bst, "btree": btree, "dict_vocab": vocab,
                    "inverted_index": inv_idx,
                })
            st.success(f"Trees built with {len(vocab)} terms.")

        bst = st.session_state["bst"]
        btree = st.session_state["btree"]
        vocab = st.session_state["dict_vocab"]
        inv_idx = st.session_state["inverted_index"]

        st.metric("Dictionary size", len(vocab))
        query_term = st.text_input("Search term in dictionary:")

        if query_term:
            q, _ = preprocess_with_config(query_term, cfg)
            q = q[0] if q else query_term.lower().strip()

            t0 = time.perf_counter()
            bst_found = bst.search(q)
            bst_search_t = (time.perf_counter() - t0) * 1e6

            t0 = time.perf_counter()
            bst_docs = retrieve_from_inverted_index(q, inv_idx) if bst_found else []
            bst_retrieval_t = (time.perf_counter() - t0) * 1e6

            t0 = time.perf_counter()
            bt_found = btree.search(q)
            bt_search_t = (time.perf_counter() - t0) * 1e6

            t0 = time.perf_counter()
            bt_docs = retrieve_from_inverted_index(q, inv_idx) if bt_found else []
            bt_retrieval_t = (time.perf_counter() - t0) * 1e6

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### BST")
                st.metric("Found", "Yes" if bst_found else "No")
                st.metric("Search time (us)", f"{bst_search_t:.2f}")
                st.metric("Retrieval time (us)", f"{bst_retrieval_t:.2f}")
                st.metric("Comparisons", bst.comparisons)
                st.write(f"Docs retrieved: {len(bst_docs)}")
            with col2:
                st.markdown("### B-Tree (t=3)")
                st.metric("Found", "Yes" if bt_found else "No")
                st.metric("Search time (us)", f"{bt_search_t:.2f}")
                st.metric("Retrieval time (us)", f"{bt_retrieval_t:.2f}")
                st.metric("Comparisons", btree.comparisons)
                st.write(f"Docs retrieved: {len(bt_docs)}")

        st.subheader("Experimental Results (Fixed Query Set)")
        bench_df, bench_summary = run_tree_benchmark(bst, btree, vocab, inv_idx)
        st.dataframe(bench_df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**BST Averages**")
            st.metric("Avg search time (us)", round(bench_summary["avg_bst_search"], 3))
            st.metric("Avg retrieval time (us)", round(bench_summary["avg_bst_retrieval"], 3))
            st.metric("Avg comparisons", round(bench_summary["avg_bst_comps"], 1))
        with col2:
            st.markdown("**B-Tree Averages**")
            st.metric("Avg search time (us)", round(bench_summary["avg_bt_search"], 3))
            st.metric("Avg retrieval time (us)", round(bench_summary["avg_bt_retrieval"], 3))
            st.metric("Avg comparisons", round(bench_summary["avg_bt_comps"], 1))

        st.success(
            f"Faster dictionary search: **{bench_summary['faster_search']}**. "
            f"Faster document retrieval: **{bench_summary['faster_retrieval']}**."
        )
        store_experiments(tree_benchmark=bench_summary, tree_df=bench_df.to_dict("records"))


# ─── TAB F ───────────────────────────────────────────────
with tabs[5]:
    st.header("F. Tolerant Retrieval")
    docs = get_docs()

    if not docs:
        st.warning("Please upload documents in Tab A first.")
    else:
        cfg = default_preprocess_config()
        all_tokens = []
        for d in docs:
            toks, _ = preprocess_with_config(d, cfg)
            all_tokens.extend(toks)
        vocabulary = sorted(set(all_tokens))
        inv_idx = build_inverted_index(docs, cfg)

        if "kgram_idx" not in st.session_state or st.session_state.get("kgram_vocab_size") != len(vocabulary):
            st.session_state["kgram_idx"] = build_kgram_index(vocabulary)
            st.session_state["kgram_vocab_size"] = len(vocabulary)

        kgram_idx = st.session_state["kgram_idx"]

        mode = st.radio(
            "Tolerant retrieval mode",
            ["K-gram Index", "Wildcard Query", "Spelling Correction (Edit Distance)", "Phonetic (Soundex)"],
            horizontal=True,
        )

        if mode == "K-gram Index":
            st.subheader("K-gram Index Representation")
            st.write(
                f"Built a **2-gram** index over **{len(vocabulary)}** vocabulary terms "
                f"with **{len(kgram_idx)}** unique k-grams. Terms are padded with `$` sentinels "
                f"(e.g. `software` → `$software$`)."
            )

            sample_rows = []
            for gram, terms in sorted(kgram_idx.items())[:12]:
                sample_rows.append({
                    "K-gram": gram,
                    "Term Count": len(terms),
                    "Sample Terms": ", ".join(sorted(terms)[:5]),
                })
            st.dataframe(pd.DataFrame(sample_rows), use_container_width=True)

            gram_query = st.text_input("Look up a k-gram:", placeholder="e.g. mi, so, ft")
            if gram_query:
                gram = gram_query.strip().lower()
                matches = lookup_kgram(gram, kgram_idx)
                st.metric("Vocabulary terms containing k-gram", len(matches))
                if matches:
                    st.success(", ".join(matches))
                    doc_ids = sorted({did for term in matches for did in inv_idx.get(term, {})})
                    st.write(f"**Documents containing matched terms ({len(doc_ids)}):**")
                    display_doc_results(doc_ids, docs, get_doc_names())
                else:
                    st.warning(f"K-gram '{gram}' not found in the index.")

            term_example = st.text_input("Show k-grams for a term:", placeholder="e.g. software")
            if term_example:
                term = term_example.strip().lower()
                if term in vocabulary:
                    padded = f"${term}$"
                    grams = [padded[i:i + 2] for i in range(len(padded) - 1)]
                    st.write(f"Padded form: `{padded}`")
                    st.code(str(grams))
                else:
                    st.error(f"Term '{term}' not in vocabulary.")

        elif mode == "Wildcard Query":
            st.subheader("Wildcard Query Search")
            st.caption("Uses the k-gram index to generate candidates, then filters with regex.")
            wq = st.text_input("Wildcard query (use *):", placeholder="e.g. soft*")
            if wq:
                t0 = time.perf_counter()
                matches = wildcard_search(wq, kgram_idx, vocabulary)
                elapsed = (time.perf_counter() - t0) * 1000
                st.metric("Matching terms", len(matches))
                st.metric("Time (ms)", f"{elapsed:.3f}")
                if matches:
                    st.write(", ".join(matches))
                    doc_ids = sorted({did for m in matches for did in inv_idx.get(m, {})})
                    st.write(f"**Documents ({len(doc_ids)}):**")
                    display_doc_results(doc_ids, docs, get_doc_names())
                else:
                    st.warning("No matching terms found.")

        elif mode == "Spelling Correction (Edit Distance)":
            misspelled = st.text_input("Misspelled term:", placeholder="e.g. retreival")
            max_d = st.slider("Max edit distance", 1, 4, 2)
            if misspelled:
                candidates = spell_correct(misspelled.lower(), vocabulary, max_dist=max_d)
                if candidates:
                    st.dataframe(pd.DataFrame(candidates, columns=["Term", "Distance"]), use_container_width=True)
                    best = candidates[0][0]
                    display_doc_results(sorted(inv_idx.get(best, {}).keys()), docs, get_doc_names())
                else:
                    st.warning("No corrections found.")
            col1, col2 = st.columns(2)
            w1 = col1.text_input("Word 1", "retrieval")
            w2 = col2.text_input("Word 2", "retreival")
            if w1 and w2:
                st.metric("Edit distance", edit_distance(w1, w2))

        else:
            phonetic_q = st.text_input("Phonetic query:", placeholder="e.g. smith")
            if phonetic_q:
                matches = phonetic_search(phonetic_q, vocabulary)
                st.metric("Soundex code", soundex(phonetic_q.upper()))
                st.metric("Matches", len(matches))
                if matches:
                    st.dataframe(pd.DataFrame([(t, soundex(t.upper())) for t in matches],
                                              columns=["Term", "Soundex"]), use_container_width=True)
                    doc_ids = sorted({did for m in matches for did in inv_idx.get(m, {})})
                    display_doc_results(doc_ids, docs, get_doc_names())

        store_experiments(tolerant_modes=["K-gram Index", "Wildcard Query", "Spelling Correction", "Phonetic (Soundex)"])


# ─── TAB G ───────────────────────────────────────────────
with tabs[6]:
    st.header("G. Inference and Discussion")
    docs = get_docs()
    exp = st.session_state.get("experiments", {})

    if docs:
        cfg = default_preprocess_config()
        tokens = []
        for d in docs:
            t, _ = preprocess_with_config(d, cfg)
            tokens.extend(t)
        c1, c2, c3 = st.columns(3)
        c1.metric("Documents", len(docs))
        c2.metric("Vocabulary", len(set(tokens)))
        c3.metric("Total Tokens", len(tokens))
    else:
        st.warning("Load documents and run experiments in other tabs for data-driven inferences.")

    best_preprocess = exp.get("best_preprocess", "Full pipeline (run Tab B)")
    stem_winner = exp.get("stem_lemma_winner", "Run Tab B experiment")
    stem_reason = exp.get("stem_lemma_reason", "")
    tree = exp.get("tree_benchmark", {})
    phrase_fp = exp.get("phrase_false_positives", "N/A")

    faster_search = tree.get("faster_search", "Run Tab E benchmark")
    faster_retrieval = tree.get("faster_retrieval", "Run Tab E benchmark")

    inferences = {
        "1. Which preprocessing technique improved retrieval quality?": (
            f"**{best_preprocess}** produced the highest total hit count across fixed test queries "
            f"({', '.join(PREPROCESS_TEST_QUERIES)}). "
            "Lowercasing normalizes case variants; stop word removal reduces noise from function words; "
            "hyphen handling merges compound tokens (e.g. anti-spyware → anti spyware). "
            "Together they improve recall and index consistency."
        ),
        "2. Was stemming or lemmatization better for this dataset?": (
            f"**{stem_winner}** performed better. {stem_reason or 'Run the Tab B comparison to populate this inference.'}"
        ),
        "3. Which phrase query index was more accurate?": (
            f"**Positional Index** is more accurate. For the last phrase query "
            f"('{exp.get('last_phrase_query', 'run Tab D')}'), biword produced "
            f"**{phrase_fp}** false positive(s) that positional search excluded. "
            "Positional verification ensures consecutive term positions."
        ),
        "4. Which tree structure was faster?": (
            f"Over {len(FIXED_BENCHMARK_QUERIES)} fixed queries: **{faster_search}** had lower average "
            f"dictionary search time; **{faster_retrieval}** had lower average document retrieval time. "
            f"BST avg comparisons: {tree.get('avg_bst_comps', 'N/A')}; "
            f"B-Tree avg comparisons: {tree.get('avg_bt_comps', 'N/A')}."
        ),
        "5. How tolerant was the retrieval model?": (
            "The system supports k-gram indexing, wildcard queries, spelling correction (Levenshtein, d≤2), "
            "and phonetic matching (Soundex). Each mode maps imperfect queries to vocabulary terms "
            "and retrieves documents via the inverted index. Tab F demonstrates all four experimentally."
        ),
        "6. What are the limitations of this system?": (
            "- In-memory indexes limit scalability.\n"
            "- BST is not self-balancing (randomized insertion mitigates skew).\n"
            "- Biword index can false-match non-contiguous phrases.\n"
            "- Soundex is coarse; Metaphone would be more accurate.\n"
            "- No ranked retrieval (TF-IDF/BM25)."
        ),
        "7. How can the system be improved?": (
            "- Add TF-IDF/BM25 ranking.\n"
            "- Use AVL/Red-Black trees or disk-backed B-Trees.\n"
            "- POS-aware lemmatization.\n"
            "- Permuterm index for wildcards.\n"
            "- Persist indexes to disk for large corpora."
        ),
    }

    for question, answer in inferences.items():
        with st.expander(question, expanded=True):
            st.markdown(answer)
