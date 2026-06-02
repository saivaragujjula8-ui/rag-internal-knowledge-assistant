import argparse

from .chunking import load_documents
from .retriever import build_index


def ingest(docs_path: str, index_path: str) -> int:
    chunks = load_documents(docs_path)
    index = build_index(chunks)
    index.save(index_path)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manual search index.")
    parser.add_argument("--docs", default="data/manuals")
    parser.add_argument("--index-path", default="indexes/manual_index.pkl")
    args = parser.parse_args()
    count = ingest(args.docs, args.index_path)
    print(f"Indexed {count} chunks")


if __name__ == "__main__":
    main()
