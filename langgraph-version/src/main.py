"""
Main entry point for Agentic Morningstar - LangGraph Version.

This implements a fully agentic research assistant with:
- Query rewriting using conversation history (from Project Morningstar app.py)
- Hybrid retrieval (vector + BM25 + fusion)
- Self-correction with retry logic
- Web fallback when local data is insufficient
"""
import os
import threading
from datetime import datetime, date
from typing import Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.logger import get_logger
logger = get_logger(__name__)

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import MorningstarState, QueryIntent, Document, SearchResult
from src.config import (
    LLM_MODEL, EMBEDDING_MODEL,
    COLLECTION_DAILY, COLLECTION_DEEP,
    CONFIDENCE_THRESHOLD, MAX_RETRIES,
    ENABLE_SMART_WEB_INGESTION, WEB_INGEST_MIN_SCORE,
    TIME_SENSITIVE_MAX_AGE_DAYS
)
from src.tools.chroma_tools import get_chroma_manager
from src.tools.search_tools import web_search_with_extraction
from src.tools.llm_tools import (
    analyze_document_relevance,
    decompose_query,
    rewrite_query_for_web,
    synthesize_answer,
    synthesize_answer_stream,
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


def decomposer_node(state: MorningstarState) -> MorningstarState:
    """
    Optionally split a complex multi-part query into atomic sub-queries.

    Simple / factual queries pass through as a single-element list so the rest
    of the pipeline is unaffected.  Complex COMPARISON / RESEARCH / EXPLORATION
    queries may become 2-3 parallel sub-queries that the librarian handles in
    separate hybrid-search calls whose results are then merged.
    """
    query = state.get("rewritten_query") or state["query"]
    intent = state["query_intent"].value if state["query_intent"] else "research"

    # Only attempt decomposition (+ keyword extraction) for multi-concept intents
    if state["query_intent"] in [QueryIntent.COMPARISON, QueryIntent.RESEARCH, QueryIntent.EXPLORATION]:
        result = decompose_query(query, intent)
    else:
        # For simple factual/summary queries still extract keywords via decompose_query
        # (it always returns a safe dict) so downstream prefilter has terms to work with
        result = decompose_query(query, intent)

    sub_queries      = result["sub_queries"]
    keywords         = result["keywords"]
    primary_terms    = result["primary_terms"]
    is_time_sensitive = result.get("is_time_sensitive", False)

    state["sub_queries"]       = sub_queries
    state["keywords"]          = keywords
    state["primary_terms"]     = primary_terms
    state["is_time_sensitive"] = is_time_sensitive

    if len(sub_queries) > 1:
        state["agent_reasoning"].append(
            f"Decomposer: Split into {len(sub_queries)} sub-queries for parallel retrieval"
        )
        for i, sq in enumerate(sub_queries, 1):
            display = sq[:70] + "..." if len(sq) > 70 else sq
            state["agent_reasoning"].append(f"  Sub-query {i}: '{display}'")
    else:
        state["agent_reasoning"].append("Decomposer: Single-concept query — no decomposition needed")

    if primary_terms:
        ts_label = " | ⏰ time-sensitive" if is_time_sensitive else ""
        state["agent_reasoning"].append(
            f"Decomposer: Keywords extracted — primary: {primary_terms}, "
            f"all: {keywords}{ts_label}"
        )

    return state


def librarian_node(state: MorningstarState) -> MorningstarState:
    """
    Query ChromaDB collections using hybrid retrieval.

    When the decomposer produced multiple sub-queries, each sub-query is run
    against both collections in parallel (ThreadPoolExecutor). Results are
    merged and deduplicated by document ID so the analyst never scores the
    same document twice.
    """
    chroma = get_chroma_manager()

    # Use sub-queries from decomposer; fall back to rewritten/original query
    sub_queries = state.get("sub_queries") or [state.get("rewritten_query") or state["query"]]
    collections = ["daily_research", "deep_dive_research"]

    intent_note = ""
    if state["query_intent"] in [QueryIntent.FACTUAL, QueryIntent.SUMMARY]:
        intent_note = " (prioritizing fast results)"
    elif state["query_intent"] == QueryIntent.RESEARCH:
        intent_note = " (deep research mode)"

    n_searches = len(sub_queries) * len(collections)
    state["agent_reasoning"].append(
        f"Librarian: Running {n_searches} hybrid searches "
        f"({len(sub_queries)} sub-quer{'y' if len(sub_queries)==1 else 'ies'} × "
        f"{len(collections)} collections){intent_note}"
    )

    def _search(query: str, collection_name: str):
        try:
            ids, documents, metadatas = chroma.hybrid_search(
                query=query,
                collection_name=collection_name
            )
            return [
                Document(id=doc_id, text=text, metadata=meta, score=0.0)
                for doc_id, text, meta in zip(ids, documents, metadatas)
            ]
        except Exception as e:
            state["agent_reasoning"].append(
                f"Librarian: Error searching '{collection_name}': {e}"
            )
            return []

    # Run all (sub_query × collection) combos in parallel
    seen_ids: dict = {}  # id → Document; first occurrence wins
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_search, q, coll): (q, coll)
            for q in sub_queries
            for coll in collections
        }
        for future in as_completed(futures):
            for doc in future.result():
                if doc["id"] not in seen_ids:
                    seen_ids[doc["id"]] = doc

    unique_docs = list(seen_ids.values())
    state["collections_tried"] = collections
    state["retrieved_docs"] = unique_docs
    state["agent_reasoning"].append(
        f"Librarian: Retrieved {len(unique_docs)} unique documents total"
    )

    return state


def analyst_node(state: MorningstarState) -> MorningstarState:
    """
    Evaluate retrieved documents and calculate confidence score.
    
    Uses LLM to score relevance 1-10 based on rubric.
    Mirrors the scoring from digest_generator.py and web_Scout.py
    """
    if not state["retrieved_docs"]:
        web_results = state.get("web_results", [])
        if web_results:
            # Web fallback already ran and returned results.
            # We have no KB docs to score, but web results ARE available for
            # synthesis — set a moderate confidence so the writer proceeds
            # and the UI shows a meaningful (not misleading 0%) confidence.
            web_confidence = 0.5  # Web-sourced answers are moderately confident
            state["confidence_score"] = web_confidence
            state["needs_web_fallback"] = False
            state["agent_reasoning"].append(
                f"Analyst: No KB docs — using {len(web_results)} web result(s), "
                f"confidence = {web_confidence:.2f}"
            )
        else:
            state["confidence_score"] = 0.0
            state["needs_web_fallback"] = True
            state["agent_reasoning"].append("Analyst: No documents retrieved, confidence = 0.0")
        return state
    
    # Score all documents in parallel (one LLM call per doc → ThreadPoolExecutor)
    query         = state.get("rewritten_query", state["query"])
    primary_terms = state.get("primary_terms", [])

    # --- Keyword pre-filter (cheap, zero LLM calls) ----------------------------
    # Match each doc's title + opening 500 chars against primary_terms.
    # Docs that don't mention ANY primary term are almost certainly irrelevant
    # and get auto-scored 1 without wasting an Ollama call.
    def _passes_prefilter(doc: dict) -> bool:
        if not primary_terms:
            return True  # No terms extracted → skip filter (safety net)
        title = doc["metadata"].get("title", "").lower()
        head  = title + " " + doc["text"][:500].lower()
        return any(term in head for term in primary_terms)

    docs_to_score   = []
    prefiltered_out = []
    for doc in state["retrieved_docs"]:
        if _passes_prefilter(doc):
            docs_to_score.append(doc)
        else:
            doc["score"] = 1
            doc["metadata"]["ai_summary"]   = "Pre-filtered: primary term not found in title/abstract."
            doc["metadata"]["ai_reasoning"] = "Skipped LLM scoring — no primary keyword match."
            prefiltered_out.append(doc)

    if prefiltered_out:
        state["agent_reasoning"].append(
            f"Analyst: Pre-filter skipped {len(prefiltered_out)} irrelevant doc(s) "
            f"(primary terms: {primary_terms})"
        )

    if not docs_to_score:
        # All docs filtered out — treat same as empty retrieval
        state["confidence_score"] = 0.0
        state["needs_web_fallback"] = True
        state["retrieved_docs"] = []
        state["agent_reasoning"].append(
            "Analyst: All docs pre-filtered out — confidence = 0.0"
        )
        return state
    # ---------------------------------------------------------------------------

    def score_doc(doc):
        analysis = analyze_document_relevance(
            title=doc["metadata"].get("title", "Untitled"),
            content=doc["text"],
            query=query
        )
        return doc, analysis

    scored_docs = []
    new_reasoning = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(score_doc, doc): doc for doc in docs_to_score}
        # Preserve original order by keying on the original doc list
        results = {}
        for future in as_completed(futures):
            doc, analysis = future.result()
            results[id(futures[future])] = (doc, analysis)

    # Rebuild in original order so reasoning steps are predictable
    for doc in docs_to_score:
        _, analysis = results[id(doc)]
        score = analysis.get("score", 0)
        doc["score"] = score
        doc["metadata"]["ai_summary"] = analysis.get("summary", "")
        doc["metadata"]["ai_reasoning"] = analysis.get("reasoning", "")
        scored_docs.append(doc)
        new_reasoning.append(
            f"Analyst: '{doc['metadata'].get('title', 'Untitled')[:40]}...' scored {score}/10"
        )

    state["agent_reasoning"].extend(new_reasoning)
    
    # Sort by score descending
    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    
    # Filter to keep only high-quality docs (score >= 7)
    high_quality = [d for d in scored_docs if d["score"] >= 7]

    # --- Staleness check for time-sensitive queries ----------------------------
    # Live data (scores, standings, prices, news) expires quickly.
    # If ALL high-quality docs are older than TIME_SENSITIVE_MAX_AGE_DAYS,
    # discard them and force a fresh web search regardless of their score.
    if high_quality and state.get("is_time_sensitive", False):
        today = date.today()
        fresh_docs = []
        stale_count = 0
        for doc in high_quality:
            date_str = doc["metadata"].get("date_ingested", "")
            try:
                doc_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                age_days = (today - doc_date).days
                if age_days <= TIME_SENSITIVE_MAX_AGE_DAYS:
                    fresh_docs.append(doc)
                else:
                    stale_count += 1
            except (ValueError, TypeError):
                # No parseable date → treat as fresh (don't penalise old-format docs)
                fresh_docs.append(doc)

        if stale_count > 0:
            state["agent_reasoning"].append(
                f"Analyst: {stale_count} high-quality doc(s) are stale "
                f"(>{TIME_SENSITIVE_MAX_AGE_DAYS}d old) for a time-sensitive query"
            )

        if not fresh_docs and stale_count > 0:
            # All high-scoring docs are stale → force web search for fresh data
            state["agent_reasoning"].append(
                "Analyst: All KB results are stale — forcing web search for live data"
            )
            state["retrieved_docs"] = []
            state["confidence_score"] = 0.0
            state["needs_web_fallback"] = True
            return state

        high_quality = fresh_docs  # keep only fresh docs for synthesis
    # ---------------------------------------------------------------------------

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
    query    = state.get("rewritten_query", state["query"])
    keywords = state.get("keywords", [])

    # Build DDG search query from pre-extracted keywords (no extra LLM call).
    # keywords were already extracted by decomposer_node, stripped of filler,
    # and capped at the most relevant terms.  Fall back to LLM rewrite only
    # when the decomposer didn't produce usable keywords (e.g., simple factual
    # queries that bypassed full decomposition or edge-case fallbacks).
    if keywords:
        web_query = " ".join(keywords[:5])   # DDG sweet-spot: ≤5 keyword tokens
        state["agent_reasoning"].append(
            f"Web Scout: Search query (from keywords) → '{web_query}'"
        )
    else:
        web_query = rewrite_query_for_web(query)
        state["agent_reasoning"].append(
            f"Web Scout: Search query (LLM rewrite) → '{web_query}'"
        )
    state["agent_reasoning"].append("Web Scout: Searching web...")

    try:
        # Search web
        results = web_search_with_extraction(
            query=web_query,
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

        # --- Primary-term filter (reuses same logic as KB prefilter) -----------
        # Drop web results whose title+snippet don't mention any primary term.
        # This removes DDG noise like off-topic blog posts, event pages, etc.
        # When primary_terms is empty (edge case) all results pass through.
        primary_terms = state.get("primary_terms", [])
        if primary_terms:
            def _web_relevant(r) -> bool:
                haystack = (r["title"] + " " + (r["snippet"] or "")).lower()
                return any(term in haystack for term in primary_terms)

            filtered   = [r for r in web_results if _web_relevant(r)]
            n_dropped  = len(web_results) - len(filtered)
            if n_dropped:
                state["agent_reasoning"].append(
                    f"Web Scout: Dropped {n_dropped} off-topic result(s) "
                    f"(primary terms: {primary_terms})"
                )
            web_results = filtered
        # -----------------------------------------------------------------------

        state["web_results"] = web_results
        state["web_search_performed"] = True
        state["web_search_count"] = state.get("web_search_count", 0) + 1

        state["agent_reasoning"].append(f"Web Scout: Found {len(web_results)} relevant web results")
        
        # Smart Ingestion: Learn from high-quality web results in a background
        # thread so the query response is never blocked by embedding calls.
        if state.get("smart_ingest_enabled", ENABLE_SMART_WEB_INGESTION) and web_results:
            web_dicts = [
                {"title": r["title"], "url": r["url"], "snippet": r["snippet"]}
                for r in web_results
            ]

            def _background_ingest(results, ctx_query):
                try:
                    ingest_web_results(
                        web_results=results,
                        query_context=ctx_query,
                        min_score=WEB_INGEST_MIN_SCORE
                    )
                except Exception:
                    pass  # Ingestion failures must never surface to the user

            thread = threading.Thread(
                target=_background_ingest,
                args=(web_dicts, web_query),  # use reformulated query as context
                daemon=True  # Dies automatically when main process exits
            )
            thread.start()
            state["agent_reasoning"].append(
                "🧠 Web Scout: Learning from web results in background..."
            )
        
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
    
    # Build citations; the actual LLM synthesis is streamed by the caller (app.py)
    citations = []
    for doc in docs_for_synthesis:
        meta = doc.get("metadata", {})
        title = meta.get("title", doc.get("id", "Unknown"))
        score_info = f" (Score: {meta.get('score', 'N/A')}/10)" if "score" in meta else ""
        citations.append(f"[{title}]({doc.get('id', '#')}){score_info}")
    if web_for_synthesis:
        for r in web_for_synthesis:
            citations.append(f"[Web: {r.get('title', 'Unknown')}]({r.get('url', '#')})")

    state["citations"] = citations
    # Empty string signals app.py to stream the synthesis
    state["synthesized_answer"] = ""

    source_type = "local DB + web" if web_for_synthesis else "local DB only"
    state["agent_reasoning"].append(
        f"Writer: Ready to synthesize using {source_type} ({len(docs_for_synthesis)} docs)"
    )
    
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

    web_search_count acts as a hard cap (max 1) so even if web results come
    back empty and analyst still flags needs_web_fallback, we never loop back
    to web_scout a second time.
    """
    web_count = state.get("web_search_count", 0)
    if state["needs_web_fallback"] and web_count < 1:
        return "web_scout"
    return "analyst"


def build_graph() -> StateGraph:
    """
    Build the LangGraph state machine.

    Flow:
    START → router → decomposer → librarian → analyst
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
                                               writer → END
    """
    workflow = StateGraph(MorningstarState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("decomposer", decomposer_node)
    workflow.add_node("librarian", librarian_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("web_scout", web_scout_node)
    workflow.add_node("writer", writer_node)

    # Add edges
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "decomposer")
    workflow.add_edge("decomposer", "librarian")
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
    logger.info("🌅 Agentic Morningstar - LangGraph Version")
    
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
        "sub_queries": [],
        "keywords": [],
        "primary_terms": [],
        "is_time_sensitive": False,
        "collections_tried": [],
        "retrieved_docs": [],
        "web_search_performed": False,
        "web_search_count": 0,
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
    
    # Run the agent (reasoning + retrieval nodes)
    config = {"configurable": {"thread_id": "test-1"}}
    result = app.invoke(initial_state, config=config)

    logger.info(f"Query: {query}")
    for i, step in enumerate(result["agent_reasoning"], 1):
        logger.info(f"  {i}. {step}")

    # Blocking synthesis for CLI (Streamlit uses the streaming path)
    docs = result.get("retrieved_docs", [])
    web = result.get("web_results") or None
    synthesis = synthesize_answer(
        query=result["query"],
        documents=docs,
        web_results=web,
        is_time_sensitive=result.get("is_time_sensitive", False)
    )
    logger.info(f"Answer: {synthesis['answer']}")
    for cite in synthesis.get('citations', []):
        logger.info(f"  • {cite}")


if __name__ == "__main__":
    main()
