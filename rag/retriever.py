"""
Retriever — finds the most relevant chunks for a query.
"""

from typing import List, Dict
from rag.embeddings import embed_query
from rag.vector_store import get_collection


def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    """
    Retrieve top_k chunks most relevant to the query.
    Returns list of dicts with 'text', 'source', 'score'.
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, dists):
        # Cosine distance → similarity score (0–1)
        score = round(1 - dist, 3)
        chunks.append(
            {
                "text": doc,
                "source": meta.get("source", "Unknown"),
                "score": score,
            }
        )

    return chunks
