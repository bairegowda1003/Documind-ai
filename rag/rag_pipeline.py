"""
Full RAG pipeline: retrieve → build prompt → generate answer.
"""

from typing import List, Dict, Tuple
from rag.retriever import retrieve
from rag.llm import call_llm

NOT_FOUND_MSG = "Information not found in uploaded documents."

SYSTEM_PROMPT = """You are DocuMind AI — an accurate, helpful academic knowledge assistant.

Your job is to answer questions using ONLY the context chunks provided below.

Rules:
1. Answer strictly from the provided context. Do NOT use outside knowledge.
2. If the answer is not in the context, reply exactly: "Information not found in uploaded documents."
3. Be concise, structured, and clear.
4. When referencing information, mention the source document name.
5. Never hallucinate or guess."""


def build_prompt_messages(
    query: str, chunks: List[Dict], chat_history: List[Dict]
) -> List[Dict[str, str]]:
    """Construct the full message list for the LLM."""
    # Build context string
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system = SYSTEM_PROMPT + f"\n\nCONTEXT:\n{context}"

    messages = [{"role": "system", "content": system}]

    # Include recent chat history (last 4 turns)
    for turn in chat_history[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": query})
    return messages


def run_rag(
    query: str, chat_history: List[Dict], top_k: int = 5
) -> Tuple[str, List[Dict]]:
    """
    Full RAG pipeline.
    Returns (answer_text, retrieved_chunks).
    """
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return NOT_FOUND_MSG, []

    messages = build_prompt_messages(query, chunks, chat_history)

    try:
        answer = call_llm(messages)
    except ValueError as e:
        return f"⚠️ {e}", chunks

    return answer, chunks
