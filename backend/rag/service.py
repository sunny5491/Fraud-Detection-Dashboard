"""
RAG Service module tying together context_builder, vector_store, and llm_client.
Provides a unified answer_question entry point for fraud investigation queries.
"""
import os
import sys
from typing import Dict, Any

# Ensure project root config is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from backend.rag import vector_store
    from backend.rag import context_builder
    from backend.rag import llm_client
except ImportError:
    try:
        from . import vector_store
        from . import context_builder
        from . import llm_client
    except ImportError:
        import vector_store
        import context_builder
        import llm_client


def answer_question(user_id: str, question: str, pipeline: Any) -> Dict[str, Any]:
    """
    Answer an investigator's question about a specific user by retrieving context,
    falling back to fresh context construction if unindexed, and generating an LLM answer.
    
    Args:
        user_id: Target user ID string.
        question: Investigator query string.
        pipeline: FraudPipeline instance.
        
    Returns:
        Dictionary: {"answer": str, "user_id": str, "sources_used": int}
    """
    if not user_id:
        return {"answer": "User not found.", "user_id": "", "sources_used": 0}

    # Requirement 2: Check if user profile exists in pipeline
    if pipeline is None:
        return {"answer": "User not found.", "user_id": user_id, "sources_used": 0}

    profile = pipeline.get_user_profile(user_id) if hasattr(pipeline, "get_user_profile") else None
    if profile is None or (isinstance(profile, dict) and "error" in profile):
        return {"answer": "User not found.", "user_id": user_id, "sources_used": 0}

    # Requirement 1: Retrieve context chunks from vector_store
    context_chunks = vector_store.retrieve_context(user_id, question)

    # If no context found, fallback to building fresh context and indexing it
    if not context_chunks:
        timeline = pipeline.get_user_timeline(user_id) if hasattr(pipeline, "get_user_timeline") else []
        # Index user case into vector store for future queries
        vector_store.index_user(user_id, profile, timeline)
        # Try retrieving context again after indexing
        context_chunks = vector_store.retrieve_context(user_id, question)

        # If Chroma retrieval still returns empty, fall back directly to fresh context text
        if not context_chunks:
            fresh_text = context_builder.build_user_context(profile, timeline)
            if fresh_text:
                context_chunks = [fresh_text]

    # Generate answer using LLM client
    answer = llm_client.generate_answer(user_id=user_id, question=question, context_chunks=context_chunks)
    sources_used = len(context_chunks)

    return {
        "answer": answer,
        "user_id": user_id,
        "sources_used": sources_used
    }
