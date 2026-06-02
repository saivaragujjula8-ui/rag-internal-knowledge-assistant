from pathlib import Path

from src.answer import answer_question
from src.ingest import ingest
from src.retriever import ManualIndex


def test_ingest_and_retrieve(tmp_path: Path) -> None:
    index_path = tmp_path / "manual_index.pkl"
    count = ingest("data/manuals", str(index_path))
    index = ManualIndex.load(index_path)
    results = index.search("What causes high turbine exhaust temperature?")

    assert count >= 2
    assert index_path.exists()
    assert results
    assert "temperature" in results[0].chunk.text.lower()


def test_answer_contains_sources(tmp_path: Path) -> None:
    index_path = tmp_path / "manual_index.pkl"
    ingest("data/manuals", str(index_path))
    results = ManualIndex.load(index_path).search("pump vibration")
    answer = answer_question("pump vibration", results)

    assert "Sources:" in answer
