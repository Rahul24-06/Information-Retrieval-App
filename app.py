
import streamlit as st
import pandas as pd
import re,time
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.metrics.distance import edit_distance

for p in ["punkt","stopwords","wordnet"]:
    nltk.download(p, quiet=True)

st.set_page_config(page_title="Group 83 IR Assignment")
st.title("Information Retrieval System - Group 83")
st.write("Starter submission package")

source = st.sidebar.radio("Dataset Source",["Upload Dataset","Load BBC Demo Dataset"])
df=None

if source=="Upload Dataset":
    f=st.file_uploader("Upload CSV",type=["csv"])
    if f: df=pd.read_csv(f)
else:
    try:
        df=pd.read_csv("datasets/bbc_news.csv")
    except:
        st.warning("Add datasets/bbc_news.csv")

if df is not None:
    st.dataframe(df.head())
    col=st.selectbox("Text Column",df.columns)
    docs=df[col].astype(str).tolist()
    stop_words=set(stopwords.words("english"))
    stemmer=PorterStemmer()
    lemmatizer=WordNetLemmatizer()
    lemma_docs=[]
    for d in docs:
        d=re.sub("-"," ",d.lower())
        toks=[t for t in word_tokenize(d) if t.isalnum() and t not in stop_words]
        lemma_docs.append([lemmatizer.lemmatize(t) for t in toks])
    inv=defaultdict(set)
    for i,tokens in enumerate(lemma_docs):
        for t in tokens:
            inv[t].add(i)
    st.write("Vocabulary Size",len(inv))
    q=st.text_input("Misspelled Term")
    if q and inv:
        best=min(inv.keys(), key=lambda x: edit_distance(q,x))
        st.success(best)
