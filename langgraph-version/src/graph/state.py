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

    # Query decomposition — complex queries are split into atomic sub-queries
    # for parallel retrieval; simple queries stay as a single-element list.
    sub_queries: List[str]

    # Keyword extraction (set by decomposer_node alongside sub_queries)
    # keywords:      all significant search terms stripped of filler words
    # primary_terms: the core must-match terms — if a doc doesn't mention any
    #                of these it is almost certainly irrelevant and can skip LLM
    keywords: List[str]
    primary_terms: List[str]

    # True when the query asks for live/current/recent data (scores, prices,
    # news, standings, etc.). Controls whether synthesis uses training knowledge
    # freely or explicitly defers to web results.
    is_time_sensitive: bool

    # Retrieval tracking
    collections_tried: List[str]
    retrieved_docs: List[Document]
    
    # Web fallback
    web_search_performed: bool
    web_search_count: int  # Guard against infinite loop: capped at 1
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
