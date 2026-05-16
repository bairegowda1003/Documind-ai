"""
Simple persistent vector store using numpy + pickle.
No ChromaDB, no opentelemetry, no protobuf — zero dependency issues.
"""

import os
import pickle
import hashlib
import numpy as np
import streamlit as st
from typing import List, Dict

STORE_PATH = "./vector_store.pkl"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a and matrix b."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b_norm @ a_norm


def _load_store() -> Dict:
    """Load store from disk or return empty store."""
    if os.path.exists(STORE_PATH):
        try:
            with open(STORE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {"documents": [], "embeddings": [], "metadatas": [], "ids": set()}


def _save_store(store: Dict) -> None:
    """Persist store to disk."""
    with open(STORE_PATH, "wb") as f:
        pickle.dump(store, f)


@st.cache_resource(show_spinner=False)
def get_store() -> Dict:
    """Return cached in-memory store, loaded from disk once."""
    return _load_store()


def add_chunks(
    chunks: List[str],
    embeddings: List[List[float]],
    source_name: str,
) -> int:
    """Add chunks to the vector store. Returns number of new chunks added."""
    store = get_store()
    added = 0

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        uid = hashlib.md5(f"{source_name}::{chunk}".encode()).hexdigest()
        if uid in store["ids"]:
            continue
        store["documents"].append(chunk)
        store["embeddings"].append(emb)
        store["metadatas"].append({"source": source_name, "chunk_index": i})
        store["ids"].add(uid)
        added += 1

    if added > 0:
        _save_store(store)

    return added


def query_store(
    query_embedding: List[float], top_k: int = 5
) -> List[Dict]:
    """Find top_k most similar chunks to the query embedding."""
    store = get_store()

    if not store["documents"]:
        return []

    emb_matrix = np.array(store["embeddings"], dtype=np.float32)
    query_vec = np.array(query_embedding, dtype=np.float32)

    scores = _cosine_similarity(query_vec, emb_matrix)
    top_k = min(top_k, len(scores))
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "text": store["documents"][idx],
            "source": store["metadatas"][idx].get("source", "Unknown"),
            "score": float(scores[idx]),
        })
    return results


def get_chunk_count() -> int:
    """Return total number of indexed chunks."""
    return len(get_store()["documents"])


def clear_collection() -> None:
    """Wipe the vector store."""
    get_store.clear()
    if os.path.exists(STORE_PATH):
        os.remove(STORE_PATH)