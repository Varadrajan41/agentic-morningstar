"""
Main entry point for Agentic Morningstar - LangGraph Version.

This implements a fully agentic research assistant with:
- Query rewriting using conversation history (from Project Morningstar app.py)
- Hybrid retrieval (vector + BM25 + fusion)
- Self-correction with retry logic
- Web fallback when local data is insufficient
"""
import os
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import MorningstarState, QueryIntent, Document, SearchResult
from src.config import (
    LLM_MODEL, EMBEDDING_MODEL,
    COLLECTION_DAILY, COLLECTION_DEEP,
    CONFIDENCE_THRESHOLD, MAX_RETRIES,
    ENABLE_SMART_WEB_INGESTION, WEB_INGEST_MIN_SCORE
)
from src.tools.chroma_tools import get_chroma_manager
from src.tools.search_tools import web_search_with_extraction
from src.tools.llm_tools import (
    analyze_document_relevance,
    synthesize_answer,
    rewrite_query_with_history,
    classify_query_intent
)
from src.ingestion.web_ingestion import ingest_web_results


def router_node(state: MorningstarState) -> MorningstarState:
    """
    Classify the query intent to determine routing.
    
    Uses LLM to determine if this is a factual lookup,
    research query, comparison, etc.
    """
    intent_str = classify_query_intent(state["query"])
    
    # Map to QueryIntent enum
    intent_map = {
        'factual': QueryIntent.FACTUAL,
        'research': QueryIntent.RESEARCH,
        'comparison': QueryIntent.COMPARISON,
        'summary': QueryIntent.SUMMARY,
        'exploration': QueryIntent.EXPLORATION
    }
    
    state["query_intent"] = intent_map.get(intent_str, QueryIntent.RESEARCH)
    state["agent_reasoning"].append(
        f"Router: Classified query as '{intent_str}' -> {state['query_intent'].value}"
    )
    
    # Rewrite query using history for better retrieval
    rewritten = rewrite_query_with_history(
        query=state["query"],
        history=state["messages"][:-1] if len(state["messages"]) > 1 else [],
        intent=intent_str
    )
    
    state["rewritten_query"] = rewritten
    state["agent_reasoning"].append(
        f"Router: Rewritten query: '{rewritten[:60]}...'" if len(rewritten) > 60 
        else f"Router: Rewritten query: '{rewritten}'"
    )
    
    return state


def librarian_node(state: MorningstarState) -> MorningstarState:
    """
    Query ChromaDB collections using hybrid retrieval.
    
    Implements: Dense embeddings + BM25 + Reciprocal Rank Fusion
    Mirrors the logic from Project Morningstar app.py
    """
    chroma = get_chroma_manager()
    
    # Use rewritten query for better retrieval
    query = state.get("rewritten_query", state["query"])
    
    # Always query both collections - learned content (web) goes to deep_dive
    # ArXiv papers go to daily_research. Intent affects ranking, not which to query.
    collections_to_query = ["daily_research", "deep_dive_research"]
    
    # Note: Intent still logged for reasoning visibility
    intent_note = ""
    if state["query_intent"] in [QueryIntent.FACTUAL, QueryIntent.SUMMARY]:
        intent_note = " (prioritizing fast results)"
    elif state["query_intent"] == QueryIntent.RESEARCH:
        intent_note = " (deep research mode)"
    
    all_retrieved = []
    
    state["agent_reasoning"].append(f"Librarian: Querying both collections{intent_note}")
    
    for collection_name in collections_to_query:
        state["agent_reasoning"].append(f"Librarian: Searching '{collection_name}'")
        
        try:
            ids, documents, metadatas = chroma.hybrid_search(
                query=query,
                collection_name=collection_name
            )
            
            # Convert to Document objects
            for doc_id, text, meta in zip(ids, documents, metadatas):
                doc = Document(
                    id=doc_id,
                    text=text,
                    metadata=meta,
                    score=0.0  # Will be filled by analyst
                )
                all_retrieved.append(doc)
            
            state["collections_tried"].append(collection_name)
            
        except Exception as e:
            state["agent_reasoning"].append(f"Librarian: Error querying {collection_name}: {e}")
    
    state["retrieved_docs"] = all_retrieved
    state["agent_reasoning"].append(f"Librarian: Retrieved {len(all_retrieved)} documents total")
    
    return state


def analyst_node(state: MorningstarState) -> MorningstarState:
    """
    Evaluate retrieved documents and calculate confidence score.
    
    Uses LLM to score relevance 1-10 based on rubric.
    Mirrors the scoring from digest_generator.py and web_Scout.py
    """
    if not state["retrieved_docs"]:
        state["confidence_score"] = 0.0
        state["needs_web_fallback"] = True
        state["agent_reasoning"].append("Analyst: No documents retrieved, confidence = 0.0")
        return state
    
    # Score each document
    scored_docs = []
    total_score = 0
    
    for doc in state["retrieved_docs"]:
        analysis = analyze_document_relevance(
            title=doc["metadata"].get("title", "Untitled"),
            content=doc["text"],
            query=state.get("rewritten_query", state["query"])
        )
        
        score = analysis.get("score", 0)
        doc["score"] = score
        doc["metadata"]["ai_summary"] = analysis.get("summary", "")
        doc["metadata"]["ai_reasoning"] = analysis.get("reasoning", "")
        
        scored_docs.append(doc)
        total_score += score
        
        state["agent_reasoning"].append(
            f"Analyst: '{doc['metadata'].get('title', 'Untitled')[:40]}...' scored {score}/10"
        )
    
    # Sort by score descending
    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    
    # Filter to keep only high-quality docs (score >= 7)
    high_quality = [d for d in scored_docs if d["score"] >= 7]
    
    # Calculate overall confidence
    if high_quality:
        avg_score = sum(d["score"] for d in high_quality) / len(high_quality)
        # Normalize to 0-1 scale
        confidence = min(avg_score / 10.0, 1.0)
    else:
        confidence = 0.0
    
    state["retrieved_docs"] = high_quality
    state["confidence_score"] = confidence
    state["needs_web_fallback"] = confidence < CONFIDENCE_THRESHOLD
    
    state["agent_reasoning"].append(
        f"Analyst: Confidence = {confidence:.2f} "
        f"({len(high_quality)} high-quality docs from {len(scored_docs)} total)"
    )
    
    if state["needs_web_fallback"]:
        state["agent_reasoning"].append(
            f"Analyst: Confidence below threshold ({CONFIDENCE_THRESHOLD}), recommending web search"
        )
    
    return state


def web_scout_node(state: MorningstarState) -> MorningstarState:
    """
    Perform web search when local data is insufficient.
    
    Uses DuckDuckGo for privacy-preserving search.
    Mirrors web_Scout.py logic.
    
    If smart ingestion is enabled, high-quality web results are
    automatically added to the knowledge base for future queries.
    """
    query = state.get("rewritten_query", state["query"])
    
    state["agent_reasoning"].append(f"Web Scout: Searching web for '{query[:50]}...'")
    
    try:
        # Search web
        results = web_search_with_extraction(
            query=query,
            max_results=5,
            extract_full=False  # Just snippets for now
        )
        
        # Convert to SearchResult format
        web_results = []
        for r in results:
            sr = SearchResult(
                title=r.get("title", "Unknown"),
                url=r.get("url", ""),
                snippet=r.get("snippet", ""),
                full_text=r.get("full_text")
            )
            web_results.append(sr)
        
        state["web_results"] = web_results
        state["web_search_performed"] = True
        
        state["agent_reasoning"].append(f"Web Scout: Found {len(web_results)} web results")
        
        # Smart Ingestion: Learn from high-quality web results
        if ENABLE_SMART_WEB_INGESTION and web_results:
            state["agent_reasoning"].append(
                f"Web Scout: Smart ingestion enabled - learning from web results..."
            )
            
            # Convert SearchResult back to dict for ingestion
            web_dicts = [
                {"title": r["title"], "url": r["url"], "snippet": r["snippet"]}
                for r in web_results
            ]
            
            # Run ingestion in background (don't block response)
            # In production, this could be async or queued
            try:
                ingest_stats = ingest_web_results(
                    web_results=web_dicts,
                    query_context=query,
                    min_score=WEB_INGEST_MIN_SCORE
                )
                
                if ingest_stats["embedded"] > 0:
                    state["agent_reasoning"].append(
                        f"🧠 Learned: {ingest_stats['embedded']} new sources added to knowledge base!"
                    )
                else:
                    state["agent_reasoning"].append(
                        f"Web Scout: Web results quality too low for knowledge base (min score: {WEB_INGEST_MIN_SCORE})"
                    )
                    
            except Exception as e:
                # Don't fail the query if ingestion fails
                state["agent_reasoning"].append(f"Web Scout: Learning skipped (error: {e})")
        
    except Exception as e:
        state["agent_reasoning"].append(f"Web Scout: Error during search: {e}")
        state["web_results"] = []
    
    return state


def writer_node(state: MorningstarState) -> MorningstarState:
    """
    Synthesize final answer from retrieved documents.
    
    Generates response with proper citations.
    """
    # Check if we should retry
    if not state["retrieved_docs"] and not state["web_results"]:
        if state["retry_count"] < MAX_RETRIES:
            state["should_retry"] = True
            state["retry_count"] += 1
            state["agent_reasoning"].append(
                f"Writer: No results available, triggering retry {state['retry_count']}/{MAX_RETRIES}"
            )
            return state
    
    state["should_retry"] = False
    
    # Prepare documents for synthesis
    docs_for_synthesis = [
        {
            "id": doc["id"],
            "text": doc["text"],
            "metadata": doc["metadata"],
            "score": doc["score"]
        }
        for doc in state["retrieved_docs"]
    ]
    
    # Prepare web results
    web_for_synthesis = None
    if state["web_results"]:
        web_for_synthesis = [
            {
                "title": r["title"],
                "url": r["url"],
                "snippet": r["snippet"]
            }
            for r in state["web_results"]
        ]
    
    # Synthesize answer
    try:
        result = synthesize_answer(
            query=state["query"],
            documents=docs_for_synthesis,
            web_results=web_for_synthesis
        )
        
        state["synthesized_answer"] = result["answer"]
        state["citations"] = result["citations"]
        
        source_type = "local DB + web" if web_for_synthesis else "local DB only"
        state["agent_reasoning"].append(
            f"Writer: Synthesized answer using {source_type} ({len(docs_for_synthesis)} docs)"
        )
        
    except Exception as e:
        state["synthesized_answer"] = f"Error generating answer: {e}"
        state["citations"] = []
        state["agent_reasoning"].append(f"Writer: Error during synthesis: {e}")
    
    return state


def should_retry(state: MorningstarState) -> Literal["librarian", "writer"]:
    """
    Conditional edge: decide whether to retry or proceed.
    """
    if state["retry_count"] < MAX_RETRIES and state["should_retry"]:
        return "librarian"
    return "writer"


def needs_web_search(state: MorningstarState) -> Literal["web_scout", "analyst"]:
    """
    Conditional edge: decide if web search is needed.
    """
    # After analyst, if we need web fallback and haven't done it yet
    if state["needs_web_fallback"] and not state["web_search_performed"]:
        return "web_scout"
    # After web_scout, we need to go back through analysis
    if state["web_search_performed"] and state["web_results"]:
        return "analyst"
    # Otherwise proceed to writer
    return "analyst"


def build_graph() -> StateGraph:
    """
    Build the LangGraph state machine.
    
    Flow:
    START -> router -> librarian -> analyst
                              |
                    (if low confidence)
                              v
                         web_scout
                              |
                              v
                    analyst (re-evaluate with web results)
                              |
                         [retry?] --yes--> librarian
                              |
                             no
                              v
                           writer -> END
    """
    workflow = StateGraph(MorningstarState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("librarian", librarian_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("web_scout", web_scout_node)
    workflow.add_node("writer", writer_node)
    
    # Add edges
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "librarian")
    workflow.add_edge("librarian", "analyst")
    
    # Conditional: web search if confidence is low
    workflow.add_conditional_edges(
        "analyst",
        needs_web_search,
        {
            "web_scout": "web_scout",
            "analyst": "writer"
        }
    )
    
    # After web scout, go back to analyst to re-evaluate with web results
    workflow.add_edge("web_scout", "analyst")
    
    # Conditional: retry if needed
    workflow.add_conditional_edges(
        "writer",
        should_retry,
        {
            "librarian": "librarian",
            "writer": END
        }
    )
    
    return workflow


def main():
    """Main entry point."""
    print("🌅 Agentic Morningstar - LangGraph Version")
    print("=" * 50)
    
    # Build graph with memory checkpointing
    workflow = build_graph()
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    
    # Example query
    query = "What are the latest advances in Agentic RAG?"
    
    # Initialize state
    initial_state: MorningstarState = {
        "query": query,
        "rewritten_query": "",
        "query_intent": None,
        "collections_tried": [],
        "retrieved_docs": [],
        "web_search_performed": False,
        "web_results": [],
        "confidence_score": 0.0,
        "needs_web_fallback": False,
        "synthesized_answer": "",
        "citations": [],
        "retry_count": 0,
        "should_retry": False,
        "smart_ingest_enabled": True,
        "messages": [{"role": "user", "content": query}],
        "agent_reasoning": []
    }
    
    # Run the agent
    config = {"configurable": {"thread_id": "test-1"}}
    result = app.invoke(initial_state, config=config)
    
    print(f"\nQuery: {query}")
    print(f"\n{'='*50}")
    print("Agent Reasoning Chain:")
    print(f"{'='*50}")
    for i, step in enumerate(result["agent_reasoning"], 1):
        print(f"  {i}. {step}")
    
    print(f"\n{'='*50}")
    print("Final Answer:")
    print(f"{'='*50}")
    print(result['synthesized_answer'])
    
    if result['citations']:
        print(f"\n{'='*50}")
        print("Sources:")
        print(f"{'='*50}")
        for cite in result['citations']:
            print(f"  • {cite}")


if __name__ == "__main__":
    main()
