"""
Retriever — finds the most relevant chunks for a query.
"""

from typing import List, Dict
from rag.embeddings import embed_query
from rag.vector_store import query_store


def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieve top_k chunks most relevant to the query.
    Returns list of dicts with 'text', 'source', 'score'.
    """
    query_embedding = embed_query(query)
    return query_store(query_embedding, top_k=top_k)