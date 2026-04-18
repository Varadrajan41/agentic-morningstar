"""
Tool implementations for agents to use.
"""
from .chroma_tools import ChromaManager, get_chroma_manager
from .search_tools import web_search, web_search_with_extraction, extract_full_text
from .llm_tools import (
    chat_with_ollama,
    analyze_document_relevance,
    synthesize_answer,
    rewrite_query_with_history,
    classify_query_intent
)

__all__ = [
    "ChromaManager",
    "get_chroma_manager",
    "web_search",
    "web_search_with_extraction",
    "extract_full_text",
    "chat_with_ollama",
    "analyze_document_relevance",
    "synthesize_answer",
    "rewrite_query_with_history",
    "classify_query_intent",
]
