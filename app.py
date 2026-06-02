from pathlib import Path

import streamlit as st

from src.answer import answer_question
from src.ingest import ingest
from src.retriever import ManualIndex


INDEX_PATH = Path("indexes/manual_index.pkl")


st.set_page_config(page_title="Internal Knowledge Assistant", page_icon="search", layout="wide")
st.title("Internal Knowledge Assistant")

with st.sidebar:
    st.header("Index")
    docs_path = st.text_input("Manuals directory", "data/manuals")
    if st.button("Rebuild index"):
        count = ingest(docs_path, str(INDEX_PATH))
        st.success(f"Indexed {count} chunks")

if not INDEX_PATH.exists():
    ingest("data/manuals", str(INDEX_PATH))

index = ManualIndex.load(INDEX_PATH)
question = st.text_input("Ask a maintenance question", "What should I check when turbine exhaust temperature is high?")

if question:
    results = index.search(question)
    st.subheader("Answer")
    st.write(answer_question(question, results))

    st.subheader("Retrieved Evidence")
    for result in results:
        with st.expander(f"{result.chunk.source} | score {result.score:.3f}"):
            st.write(result.chunk.text)
