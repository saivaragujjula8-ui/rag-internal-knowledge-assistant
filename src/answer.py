import os

from dotenv import load_dotenv

from .retriever import RetrievedChunk


def build_extractive_answer(question: str, results: list[RetrievedChunk]) -> str:
    if not results:
        return "I could not find relevant maintenance guidance in the indexed manuals."
    best = results[0].chunk
    citations = ", ".join(f"{item.chunk.source} ({item.score:.2f})" for item in results)
    return (
        f"Based on the manuals, the most relevant guidance is: {best.text}\n\n"
        f"Sources: {citations}\n\n"
        f"Question answered: {question}"
    )


def answer_question(question: str, results: list[RetrievedChunk]) -> str:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        return build_extractive_answer(question, results)
    return build_extractive_answer(question, results)
