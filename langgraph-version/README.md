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
| **Query Intent Classification** | LLM classifies query as factual/research/comparison/etc. |
| **Autonomous Collection Selection** | Agent decides which collections to query based on intent |
| **Self-Correction** | Retry mechanism when no results found |
| **Agent Reasoning Chain** | Full visibility into agent decision-making |
| **State Machine** | Explicit graph with conditional edges |
| **Checkpointing** | Resume from crashes with `MemorySaver` |

## Architecture

```
                    START
                      │
                ┌─────▼─────┐
                │   Router  │ ← Query intent classification
                │           │ ← Query rewriting with history
                └─────┬─────┘
                      │
                      ▼
                ┌─────────────┐
                │  Librarian  │ ← Hybrid retrieval (vector + BM25)
                │             │ ← Decides collections by intent
                └─────┬───────┘
                      │
                      ▼
                ┌─────────────┐
                │   Analyst   │ ← LLM relevance scoring 1-10
                │             │ ← Calculates confidence
                └─────┬───────┘
                      │
           ┌──────────┴──────────┐
           │  (if confidence < 0.7) │
           ▼                      ▼
    ┌──────────────┐      ┌─────────────┐
    │   Web Scout  │      │    Writer   │
    │  (DDGS search)│      │  (synthesis) │
    └──────┬───────┘      └──────┬──────┘
           │                     │
           └──────────┬──────────┘
                      │
                 ┌────▼────┐
                 │   END   │
                 └─────────┘
```

## Files

```
langgraph-version/
├── app.py                      # Streamlit UI (like original app.py)
├── requirements.txt            # Dependencies
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration (models, thresholds)
│   ├── main.py                # LangGraph builder + agent nodes
│   ├── graph/
│   │   ├── __init__.py
│   │   └── state.py           # MorningstarState TypedDict
│   ├── agents/                # (reserved for future split)
│   └── tools/
│       ├── __init__.py
│       ├── chroma_tools.py    # ChromaDB + BM25 + RRF
│       ├── search_tools.py    # DDGS web search
│       └── llm_tools.py       # Ollama integration
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

### 3. Agentic Decision Making

From `src/main.py`:

```python
def librarian_node(state: MorningstarState):
    # Agent decides which collections to query
    if state["query_intent"] == QueryIntent.FACTUAL:
        collections = ["daily_research"]  # Quick lookup
    elif state["query_intent"] == QueryIntent.RESEARCH:
        collections = ["daily_research", "deep_dive_research"]  # Deep search
    
    # Query selected collections
    for coll in collections:
        results = chroma.hybrid_search(query, collection_name=coll)
        ...
```

### 4. Web Fallback with Confidence Threshold

From `src/main.py`:

```python
def analyst_node(state: MorningstarState):
    # Score documents
    confidence = sum(scores) / len(scores) / 10.0
    state["confidence_score"] = confidence
    
    # Trigger web search if below threshold
    state["needs_web_fallback"] = confidence < CONFIDENCE_THRESHOLD

def needs_web_search(state) -> Literal["web_scout", "analyst"]:
    if state["needs_web_fallback"] and not state["web_search_performed"]:
        return "web_scout"  # Trigger web search
    return "analyst"  # Skip to writer
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
🌅 Agentic Morningstar - LangGraph Version
==================================================

Query: What are the latest advances in Agentic RAG?

==================================================
Agent Reasoning Chain:
==================================================
  1. Router: Classified query as 'research'
  2. Router: Rewritten query: 'latest advances in Agentic RAG 2026...'
  3. Librarian: Querying 'daily_research' collection
  4. Librarian: Querying 'deep_dive_research' collection
  5. Librarian: Retrieved 6 documents total
  6. Analyst: 'Paper 1: Multi-Agent...' scored 8/10
  7. Analyst: 'Paper 2: LangGraph...' scored 9/10
  8. Analyst: Confidence = 0.75 (5 high-quality docs)
  9. Writer: Synthesized answer using local DB only (5 docs)

==================================================
Final Answer:
==================================================
Based on the retrieved research papers...
```

## Next Steps / TODO

- [ ] Add human-in-the-loop with `interrupt_before`
- [ ] Implement parallel collection querying
- [ ] Add ArXiv ingestion (from digest_generator.py)
- [ ] Add query decomposition for complex questions
- [ ] Persistent storage for thread history
- [ ] Better error handling and retry logic
