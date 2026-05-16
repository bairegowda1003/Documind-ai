"""
ChromaDB vector store — persistent, cached.
"""

import streamlit as st
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any

COLLECTION_NAME = "documind_knowledge"
CHROMA_PATH = "./chroma_db"


@st.cache_resource(show_spinner=False)
def get_chroma_client() -> chromadb.PersistentClient:
    """Return a cached ChromaDB persistent client."""
    return chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


@st.cache_resource(show_spinner=False)
def get_collection():
    """Return cached ChromaDB collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(
    chunks: List[str],
    embeddings: List[List[float]],
    source_name: str,
) -> int:
    """
    Add chunks + embeddings to the collection.
    Skips duplicates by using source+index as ID.
    Returns number of new chunks added.
    """
    collection = get_collection()

    # Build IDs based on content hash to avoid duplicates
    import hashlib

    ids = [
        hashlib.md5(f"{source_name}::{chunk}".encode()).hexdigest()
        for chunk in chunks
    ]
    metadatas = [{"source": source_name, "chunk_index": i} for i, _ in enumerate(chunks)]

    # Filter out already-existing IDs
    existing = set(collection.get(ids=ids)["ids"])
    new_indices = [i for i, uid in enumerate(ids) if uid not in existing]

    if not new_indices:
        return 0

    collection.add(
        documents=[chunks[i] for i in new_indices],
        embeddings=[embeddings[i] for i in new_indices],
        ids=[ids[i] for i in new_indices],
        metadatas=[metadatas[i] for i in new_indices],
    )
    return len(new_indices)


def get_chunk_count() -> int:
    """Return total number of indexed chunks."""
    return get_collection().count()


def clear_collection() -> None:
    """Delete and recreate the collection."""
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    # Clear caches so the new collection is picked up
    get_collection.clear()
