"""
Vector Store module for indexing and retrieving user case contexts using SentenceTransformers and ChromaDB.
"""
import os
import sys
from typing import List, Dict, Optional, Any

import chromadb
from sentence_transformers import SentenceTransformer

# Ensure project root config is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    import config
except ImportError:
    sys.path.append(os.getcwd())
    import config

try:
    from backend.rag.context_builder import build_user_context, chunk_context
except ImportError:
    try:
        from .context_builder import build_user_context, chunk_context
    except ImportError:
        from context_builder import build_user_context, chunk_context

_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None


def get_embedding_model() -> SentenceTransformer:
    """
    Get or initialize the shared SentenceTransformer model instance.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedding_model


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Get or initialize the shared ChromaDB PersistentClient instance.
    """
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    return _chroma_client


def get_collection():
    """
    Get or create the 'user_cases' collection in ChromaDB.
    """
    client = get_chroma_client()
    return client.get_or_create_collection(name="user_cases")


def index_user(user_id: str, profile: dict, timeline: list) -> None:
    """
    Build context text for a user, chunk it, embed each chunk, and upsert into ChromaDB.
    
    Args:
        user_id: User identifier string.
        profile: Dictionary containing user risk scores, features, and SHAP data.
        timeline: List of transaction/return records for the user.
    """
    if not user_id or not profile:
        return

    context = build_user_context(profile, timeline)
    chunks = chunk_context(context)

    if not chunks:
        return

    documents = []
    metadatas = []
    ids = []

    if isinstance(chunks, dict):
        items = list(chunks.items())
    elif isinstance(chunks, list):
        items = []
        for idx, item in enumerate(chunks):
            if isinstance(item, dict):
                items.append((item.get("section", f"section_{idx}"), item.get("text", "")))
            elif isinstance(item, tuple) and len(item) == 2:
                items.append(item)
            elif isinstance(item, str):
                items.append((f"section_{idx}", item))
    else:
        items = [("general", str(chunks))]

    for section, text in items:
        if not text or not str(text).strip():
            continue
        sec_str = str(section)
        doc_id = f"{user_id}_{sec_str}"
        documents.append(str(text))
        metadatas.append({"user_id": str(user_id), "section": sec_str})
        ids.append(doc_id)

    if not documents:
        return

    model = get_embedding_model()
    embeddings = model.encode(documents).tolist()

    collection = get_collection()
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )


def index_all_users(pipeline: Any) -> None:
    """
    Loop over all users in pipeline.user_features and call index_user for each.
    Called once after the ML pipeline runs.
    
    Args:
        pipeline: FraudPipeline instance containing trained user features.
    """
    if pipeline is None:
        return

    user_features = getattr(pipeline, "user_features", None)
    if user_features is None or getattr(user_features, "empty", True):
        return

    if "user_id" not in user_features.columns:
        return

    for user_id in user_features["user_id"].dropna().unique():
        user_id_str = str(user_id)
        profile = pipeline.get_user_profile(user_id_str)
        timeline = pipeline.get_user_timeline(user_id_str)
        if profile:
            index_user(user_id=user_id_str, profile=profile, timeline=timeline)


def retrieve_context(user_id: str, query: str, top_k: int = 3) -> list[str]:
    """
    Retrieve top_k most relevant context chunks for a specific user_id query.
    Filters Chroma query to metadata user_id == user_id so context from other users is never leaked.
    
    Args:
        user_id: User identifier to query context for.
        query: Investigator's search query string.
        top_k: Maximum number of relevant chunks to retrieve.
        
    Returns:
        List of matching context chunk texts, or empty list if no matches or collection is empty/uninitialized.
    """
    if not user_id or not query:
        return []

    try:
        collection = get_collection()
        if collection.count() == 0:
            return []

        model = get_embedding_model()
        query_embedding = model.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where={"user_id": str(user_id)}
        )

        documents = results.get("documents", [])
        if documents and len(documents) > 0 and isinstance(documents[0], list):
            return documents[0]

        return []
    except Exception:
        # Handle empty collection, missing model, or unindexed user gracefully
        return []
