# Agentic Morningstar - LangGraph Version

A fully agentic research assistant built with LangGraph, extending the original Project Morningstar with true agentic decision-making.

## What This Implements

This is a **complete, working implementation** of all Project Morningstar features using LangGraph:

### ✅ Features from Original Project Morningstar

| Feature | Original (app.py) | This Implementation |
|---------|-------------------|---------------------|
| **Query Rewriting** | Lines 95-107: Uses conversation history to reformulate queries | ✅ `router_node()` uses `rewrite_query_with_history()` |
| **Tiered Collections** | Two collections: `daily_research`, `deep_dive_research` | ✅ Same collections, agent chooses which to query |
| **Hybrid Retrieval** | Vector + BM25 + Reciprocal Rank Fusion | ✅ `ChromaManager.hybrid_search()` |
| **Web Fallback** | DDGS search when local results insufficient | ✅ `web_scout_node()` with conditional edge |
| **LLM Scoring** | 1-10 relevance scoring with rubric | ✅ `analyze_document_relevance()` |
| **Response Synthesis** | Generate answer with citations | ✅ `writer_node()` with `synthesize_answer()` |
| **Streamlit UI** | Two toggle modes, chat interface | ✅ `app.py` with reasoning display |
| **Local-First** | Ollama + ChromaDB | ✅ Same stack |

### ✅ New Agentic Features

| Feature | Description |
|---------|-------------|
| **Query Intent Classification** | LLM classifies query as factual/research/comparison/summary/exploration |
| **Query Decomposition** | Complex multi-concept queries split into 2–3 parallel sub-queries |
| **Keyword Extraction** | `primary_terms` extracted for pre-filtering; `keywords` used for web search |
| **Time-Sensitivity Detection** | Live queries (scores, prices, news) force a web search if KB docs are stale |
| **Parallel Hybrid Search** | All sub-query × collection combos run concurrently (ThreadPoolExecutor) |
| **Keyword Pre-filter** | Docs not mentioning any primary term skip LLM scoring entirely |
| **Staleness Check** | KB docs older than `TIME_SENSITIVE_MAX_AGE_DAYS` discarded for live queries |
| **Smart Web Ingestion** | High-quality web results auto-ingested into KB in a background thread |
| **ArXiv Daily Ingestion** | Pipeline to fetch, score, and embed ArXiv papers on a schedule |
| **RSS Feed Subscriptions** | Subscribe to any RSS/Atom feed; auto-detect topic context via LLM |
| **Self-Correction** | Retry mechanism (up to 2 retries) when no results found |
| **Agent Reasoning Chain** | Full visibility into every decision the agent made |
| **State Machine** | Explicit graph with typed state and conditional edges |
| **Checkpointing** | Resume from crashes with `MemorySaver` |

## Architecture

```
                    START
                      │
                ┌─────▼─────┐
                │   Router  │ ← Intent classification + query rewrite with history
                └─────┬─────┘
                      │
                      ▼
                ┌─────────────┐
                │  Decomposer │ ← Split into sub-queries; extract keywords & primary_terms
                │             │ ← Detect time-sensitivity
                └─────┬───────┘
                      │
                      ▼
                ┌─────────────┐
                │  Librarian  │ ← Parallel hybrid search (all sub-queries × both collections)
                │             │ ← Vector + BM25 + RRF; deduplicate by doc ID
                └─────┬───────┘
                      │
                      ▼
                ┌─────────────┐
                │   Analyst   │ ← Keyword pre-filter → LLM score 1-10 → staleness check
                │             │ ← Confidence score; flags needs_web_fallback
                └─────┬───────┘
                      │
        ┌─────────────┴──────────────┐
        │  confidence < 0.6          │  confidence ≥ 0.6
        ▼                            ▼
┌──────────────┐             ┌─────────────┐
│   Web Scout  │             │    Writer   │ ← Stream answer + citations
│  DuckDuckGo  │             └──────┬──────┘
│  + smart     │                    │
│  ingestion   │             ┌──────┴──────┐
└──────┬───────┘             │  no results?│
       │                     │  retry (×2) → Librarian
       └──► Analyst           │
            (re-eval)         └──► END
```

## Files

```
langgraph-version/
├── app.py                      # Streamlit UI with streaming synthesis + reasoning trace
├── requirements.txt            # Dependencies
├── rss_feeds.json              # Persisted RSS feed subscriptions
├── IMPROVEMENTS.md             # Changelog and design decision log
├── src/
│   ├── __init__.py
│   ├── config.py              # All settings (env-overridable)
│   ├── main.py                # LangGraph graph definition + all node functions
│   ├── graph/
│   │   ├── __init__.py
│   │   └── state.py           # MorningstarState TypedDict + QueryIntent enum
│   ├── agents/                # (reserved)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── chroma_tools.py    # ChromaManager: hybrid search + BM25 + RRF
│   │   ├── search_tools.py    # DuckDuckGo search + trafilatura extraction
│   │   └── llm_tools.py       # All Ollama interactions (score, decompose, synthesize, stream)
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── arxiv_fetcher.py   # ArXiv daily ingestion pipeline
│   │   ├── web_ingestion.py   # Smart web ingestion + manual topic ingest
│   │   ├── rss_fetcher.py     # RSS/Atom feed subscription + ingestion
│   │   └── backup.py          # KB backup utilities
│   └── utils/
│       └── logger.py
└── README.md
```

## Key Components

### 1. Query Rewriting with History

From `src/tools/llm_tools.py`:

```python
def rewrite_query_with_history(query: str, history: list) -> str:
    """
    Rewrites query using conversation history.
    Mirrors app.py lines 95-107 from original Project Morningstar.
    """
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])
    
    system_prompt = "Rewrite the user's latest query into a standalone search query..."
    
    rewritten = chat_with_ollama(system_prompt, user_prompt)
    return rewritten
```

### 2. Hybrid Retrieval (Vector + BM25 + Fusion)

From `src/tools/chroma_tools.py`:

```python
def hybrid_search(self, query: str, collection_name: str):
    # 1. Vector search
    embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
    vector_results = collection.query(query_embeddings=[embedding], n_results=5)
    
    # 2. BM25 search
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked = sorted(zip(ids, bm25_scores), key=lambda x: x[1], reverse=True)
    
    # 3. Reciprocal Rank Fusion
    fused = self.reciprocal_rank_fusion(vector_ids, bm25_ids)
    
    return fused_results
```

### 3. Query Decomposition + Parallel Retrieval

From `src/main.py`:

```python
def decomposer_node(state):
    result = decompose_query(query, intent)
    # result = {sub_queries, keywords, primary_terms, is_time_sensitive}
    state["sub_queries"] = result["sub_queries"]       # 1–3 atomic queries
    state["primary_terms"] = result["primary_terms"]   # used for pre-filtering
    state["is_time_sensitive"] = result["is_time_sensitive"]

def librarian_node(state):
    # Runs all sub-query × collection combos in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_search, q, coll): (q, coll)
            for q in state["sub_queries"]
            for coll in ["daily_research", "deep_dive_research"]
        }
    # Deduplicate across all results by doc ID
```

### 4. Web Fallback with Confidence Threshold

From `src/main.py`:

```python
def analyst_node(state):
    # Keyword pre-filter (zero LLM calls for obvious mismatches)
    docs_to_score = [d for d in docs if _passes_prefilter(d)]
    
    # LLM scoring in parallel
    high_quality = [d for d in scored_docs if d["score"] >= 7]
    confidence = avg(scores) / 10.0
    state["needs_web_fallback"] = confidence < CONFIDENCE_THRESHOLD  # 0.6

def needs_web_search(state) -> Literal["web_scout", "analyst"]:
    # Hard cap: at most 1 web search per query (prevents loops)
    if state["needs_web_fallback"] and state["web_search_count"] < 1:
        return "web_scout"
    return "analyst"
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run CLI version
python -m src.main

# Run Streamlit UI
streamlit run app.py
```

## Requirements

- Python 3.10+
- Ollama running locally with:
  - `qwen2.5:7b-instruct` (LLM)
  - `nomic-embed-text` (embeddings)
- ChromaDB (installed via pip)

## Differences from Original

| Aspect | Original | This Version |
|--------|----------|--------------|
| **Architecture** | Monolithic scripts | Agentic state machine |
| **Query Routing** | Manual toggle | Automatic by intent |
| **Retrieval** | Pre-configured | Agent decides collections |
| **Web Fallback** | Automatic on no results | Confidence-based decision |
| **Observability** | Limited | Full reasoning chain |
| **Extensibility** | Hard to modify | Easy to add nodes/edges |

## Example Output

```
Query: Compare SGLang vs vLLM for LLM inference

Agent Reasoning Chain:
  1. Router: Classified query as 'comparison' -> comparison
  2. Router: Rewritten query: 'SGLang vs vLLM LLM inference throughput latency'
  3. Decomposer: Split into 2 sub-queries for parallel retrieval
  4.   Sub-query 1: 'SGLang LLM inference throughput performance'
  5.   Sub-query 2: 'vLLM continuous batching latency benchmark'
  6. Decomposer: Keywords extracted — primary: ['sglang', 'vllm'], all: [...]
  7. Librarian: Running 4 hybrid searches (2 sub-queries × 2 collections)
  8. Librarian: Retrieved 9 unique documents total
  9. Analyst: Pre-filter skipped 3 irrelevant doc(s) (primary terms: ['sglang', 'vllm'])
 10. Analyst: 'SGLang: Efficient Execution of Structured...' scored 9/10
 11. Analyst: 'vLLM: Easy, Fast, and Cheap LLM Serving...' scored 9/10
 12. Analyst: Confidence = 0.88 (4 high-quality docs from 6 total)
 13. Writer: Ready to synthesize using local DB only (4 docs)

Final Answer:
Based on the retrieved papers, SGLang and vLLM differ primarily in...
  • [SGLang: Efficient Execution...](arxiv.org/...) (Score: 9/10)
  • [vLLM: Easy, Fast and Cheap...](arxiv.org/...) (Score: 9/10)
```

## Next Steps / TODO

- [x] Implement parallel collection querying (ThreadPoolExecutor)
- [x] Add ArXiv ingestion pipeline (`src/ingestion/arxiv_fetcher.py`)
- [x] Add query decomposition for complex questions (`decomposer_node`)
- [x] Better error handling and retry logic (up to 2 retries via `should_retry` edge)
- [x] Smart web ingestion — auto-learn from web results in background thread
- [x] RSS feed ingestion with auto-detected topic context
- [x] Time-sensitive query detection and staleness-based web fallback
- [x] Keyword pre-filter to skip LLM scoring on obvious mismatches
- [ ] Add human-in-the-loop with `interrupt_before`
- [ ] Persistent storage for thread history (swap `MemorySaver` → `PostgresSaver`)
- [ ] CrewAI version for framework comparison
