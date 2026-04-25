"""
State definitions for the Agentic Morningstar LangGraph.
"""
from typing import TypedDict, List, Dict, Any, Optional
from enum import Enum


class QueryIntent(str, Enum):
    """Classification of query types for routing decisions."""
    FACTUAL = "factual"           # Simple factual lookup
    RESEARCH = "research"         # Deep research query
    COMPARISON = "comparison"     # Compare multiple items
    SUMMARY = "summary"           # Summarize from sources
    EXPLORATION = "exploration" # Open-ended exploration


class Document(TypedDict):
    """A retrieved document with metadata."""
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class SearchResult(TypedDict):
    """A web search result."""
    title: str
    url: str
    snippet: str
    full_text: Optional[str]


class MorningstarState(TypedDict):
    """
    Shared state that flows through the LangGraph.
    This is the central data structure for all agents.
    """
    # Input
    query: str
    rewritten_query: str  # Query after history-based reformulation
    query_intent: Optional[QueryIntent]
    
    # Retrieval tracking
    collections_tried: List[str]
    retrieved_docs: List[Document]
    
    # Web fallback
    web_search_performed: bool
    web_results: List[SearchResult]
    
    # Evaluation
    confidence_score: float
    needs_web_fallback: bool
    
    # Smart ingestion
    smart_ingest_enabled: bool  # Whether to auto-learn from web results
    
    # Synthesis
    synthesized_answer: str
    citations: List[str]
    
    # Control
    retry_count: int
    should_retry: bool
    
    # Messages for UI display
    messages: List[Dict[str, str]]  # [{"role": "user|assistant", "content": "..."}]
    agent_reasoning: List[str]  # Step-by-step agent thoughts
