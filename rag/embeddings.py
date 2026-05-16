"""
Sentence-transformer embedding model loader (cached).
"""

import streamlit as st
from sentence_transformers import SentenceTransformer
from typing import List

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def get_embed_model() -> SentenceTransformer:
    """Load and cache embedding model once per session."""
    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of text strings."""
    model = get_embed_model()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    model = get_embed_model()
    return model.encode([query])[0].tolist()
