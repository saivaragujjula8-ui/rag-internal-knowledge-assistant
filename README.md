# RAG-Based Internal Knowledge Assistant

Retrieval-augmented generation assistant for maintenance engineers searching thousands of PDF manuals.

This repo includes local ingestion, vector retrieval, a Streamlit UI, tests, and deployment files. It is designed to work without paid credentials in demo mode and can be connected to OpenAI for production responses.

## Features

- PDF/text manual ingestion
- Chunking and metadata capture
- FAISS vector search when installed
- NumPy vector search fallback for simple demos
- Streamlit web application
- Optional OpenAI answer generation
- Tests, Dockerfile, and GitHub Actions CI

## Architecture

```text
PDF Manuals -> Text Extraction -> Chunking -> Embeddings -> FAISS Index -> Retriever -> Assistant UI
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.ingest --docs data/manuals --index-path indexes/manual_index.pkl
streamlit run app.py
```

Run tests:

```bash
pytest
```

## Configuration

Copy `.env.example` to `.env` for OpenAI-backed answers.

```text
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
```

Without an API key, the app returns extractive answers from retrieved manual passages.

## Repository Structure

```text
app.py
src/
  answer.py
  chunking.py
  embeddings.py
  ingest.py
  retriever.py
data/manuals/
indexes/
tests/
```


