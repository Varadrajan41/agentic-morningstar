# Project Morningstar: Agentic Research Assistant

> A fully local, self-correcting research assistant built on LangGraph.  
> Think of it as a personal Perplexity + Elicit hybrid that runs entirely on your machine, continuously learns from what it searches, and shows you its reasoning step by step.

---

## What Is This?

Project Morningstar is an **agentic RAG (Retrieval-Augmented Generation) system** — a research assistant that doesn't just look things up, it *reasons* about whether what it found is good enough, decides autonomously to search the web when it isn't, and gets smarter after every query by learning from what it discovers.

It was built by studying the original Project Morningstar (a static hybrid RAG app) and rebuilding it as a true agentic system — with a state machine, self-correction, and continuous learning.

**Status:** LangGraph version is fully implemented and working. A CrewAI version is planned for comparison.

---

## Who Is This For?

### Primary: AI/ML Researcher or Practitioner

**Problem:** ArXiv publishes hundreds of papers a day. There is no personal assistant that (a) automatically curates the relevant ones into a local knowledge base, (b) answers research questions conversationally, and (c) supplements gaps with fresh web searches — all without sending your data to OpenAI or any cloud API.

**What this delivers:** A personal research assistant that knows the papers you care about, runs on your laptop, and gets smarter every time you use it.

**Signs this is you:**
- You track AI/ML or cybersecurity research regularly
- You want to ask questions across a growing private knowledge base
- You care about reproducibility — you want to see *why* the system gave you that answer

---

### Secondary: Privacy-Conscious Knowledge Worker in a Technical Domain

**Problem:** Deep research often involves sensitive topics (proprietary tech, internal systems, security vulnerabilities). Sending queries to cloud LLM APIs is not acceptable.

**What this delivers:** Everything — LLM inference, embeddings, vector DB, web search — runs locally. No data leaves your machine. Subscribe your own RSS feeds and build a curated knowledge base on any private topic.

---

### Tertiary: Developer / AI Engineer Learning Agentic RAG Patterns

**Problem:** Most RAG tutorials are toys. There is no production-grade reference implementation of a fully agentic RAG pipeline that shows self-correction, hybrid retrieval, query decomposition, and continuous learning all working together.

**What this delivers:** A clean, end-to-end reference architecture for:
- LangGraph state machine design with typed state
- Confidence-gated web fallback
- Background smart ingestion
- Streaming synthesis in Streamlit

---

## What It Is NOT

- **Not a general-purpose chatbot** — it only knows what is in its local KB (plus what it finds on the web when needed)
- **Not enterprise-grade** — single-user, no access control, no distributed infrastructure
- **Not a real-time monitor** — it learns from web results when you query, not from a background crawler
- **Not cloud-dependent** — everything runs on one machine; no OpenAI, no cloud vector DB, no paid APIs

---

## The LangGraph Version (Implemented)

### Architecture: The Pipeline

```
START
  │
  ▼
Router          — Classifies query intent; rewrites query using conversation history
  │
  ▼
Decomposer      — Splits complex multi-concept queries into 2–3 parallel sub-queries;
  │               extracts keywords and primary terms; detects time-sensitivity
  ▼
Librarian       — Hybrid search (vector + BM25 + RRF) across both KB collections,
  │               in parallel for every sub-query × collection combination
  ▼
Analyst         — Keyword pre-filter → LLM scores each doc 1–10 → staleness check
  │               for time-sensitive queries → computes confidence score
  │
  ├─── confidence ≥ 0.6 ──────────────────────────────────────┐
  │                                                             │
  └─── confidence < 0.6 ───► Web Scout                        │
                               │  DuckDuckGo search            │
                               │  Primary-term filter          │
                               │  Smart ingestion (background) │
                               └──────────► Analyst (re-eval)  │
                                                                │
                                                                ▼
                                                             Writer
                                                               │
                                                 ┌── no results? retry → Librarian
                                                 │
                                                 └── results? → END (stream answer)
```

### Agent Nodes

| Node | Role |
|------|------|
| **Router** | Classifies intent (factual / research / comparison / summary / exploration). Rewrites query using last 5 messages of conversation history so follow-ups work correctly. |
| **Decomposer** | Breaks multi-concept queries into up to 3 atomic sub-queries. Extracts all keywords and a strict `primary_terms` subset used for pre-filtering. Flags time-sensitive queries. |
| **Librarian** | Runs hybrid search across `daily_research` and `deep_dive_research` in parallel (ThreadPoolExecutor). Deduplicates by document ID across all sub-query × collection results. |
| **Analyst** | Pre-filters with keyword matching (zero LLM calls), then LLM-scores remaining docs 1–10. Discards stale KB docs for time-sensitive queries and forces web search. Computes overall confidence. |
| **Web Scout** | DuckDuckGo search using pre-extracted keywords (no extra LLM call). Drops off-topic results. Triggers background smart ingestion of high-quality results. |
| **Writer** | Prepares citations; signals the Streamlit app to stream the synthesis. Triggers retry via `librarian` if no results are available (up to 2 retries). |

### Hybrid Retrieval (Librarian + ChromaDB)

Every search is three searches fused:

1. **Vector search** — semantic similarity via `nomic-embed-text` embeddings (top-K=5)
2. **BM25 keyword search** — exact/partial keyword matching using `rank-bm25` (top-K=5)
3. **Reciprocal Rank Fusion (RRF)** — combines both ranked lists into a single fused ranking (k=60, final top 6 candidates passed to the Analyst)

Two ChromaDB collections:

| Collection | Content |
|-----------|---------|
| `daily_research` | ArXiv papers (ingested daily, scored ≥ 7/10) |
| `deep_dive_research` | Web content from smart ingestion and RSS feeds (scored ≥ 8/10) |

### Smart Ingestion (Continuous Learning)

The system gets smarter automatically in three ways:

**1. Smart Web Ingestion (query-triggered)**  
When the Web Scout runs, high-quality results are ingested into `deep_dive_research` in a background daemon thread — the user gets their answer immediately, and the KB improves for next time.

**2. Daily ArXiv Ingestion (scheduled)**  
The ingestion pipeline fetches papers from ArXiv (configurable query, default `cat:cs.AI OR cat:cs.CR`), scores each paper's abstract with the LLM (1–10), and embeds papers scoring ≥ 7 into `daily_research`. Already-ingested papers are skipped to avoid redundant LLM calls.

**3. RSS Feed Subscriptions**  
Subscribe to any Atom/RSS feed. Each entry is fetched, full text is extracted with `trafilatura`, scored by the LLM for relevance, and embedded if score ≥ 7. The system auto-detects the feed's topic context via LLM so you don't need to configure it manually. Feed subscriptions persist in `rss_feeds.json`.

### LLM Document Scoring

Every candidate document is scored against a strict rubric:

- The **title** is the primary indicator — a document whose title is about a different concept gets score 1–3 even if the query is mentioned in the body
- Auto-correction: if the LLM gives a high score but keywords are absent from the title, the score is capped at 3–4
- Only documents scoring ≥ 7 are kept for synthesis

### Time-Sensitive Query Handling

When the Decomposer flags a query as time-sensitive (keywords: "current", "today", "standings", "price", "live", "score", specific recent years):
- KB documents older than `TIME_SENSITIVE_MAX_AGE_DAYS` (default: 1 day) are considered stale
- If all high-scoring KB docs are stale, they are discarded and a web search is forced regardless of confidence score
- During synthesis, the LLM is explicitly instructed to treat web results as authoritative over its training knowledge

### Answer Synthesis

- **Streaming** (Streamlit UI): token-by-token via Ollama's streaming chat API, with a live `▌` cursor
- **Blocking** (CLI): standard chat completion
- Citations from both KB documents and web results are included inline
- Temporal mode: time-sensitive queries get a different system prompt that tells the LLM to defer to web results

### Transparent Agent Reasoning

Every step appends a human-readable line to `agent_reasoning`. The Streamlit UI displays this trace so you can see exactly why the system did what it did:

```
1. Router: Classified query as 'comparison' -> comparison
2. Router: Rewritten query: 'SGLang vs vLLM throughput latency comparison...'
3. Decomposer: Split into 2 sub-queries for parallel retrieval
4.   Sub-query 1: 'SGLang LLM inference throughput performance'
5.   Sub-query 2: 'vLLM continuous batching latency'
6. Decomposer: Keywords extracted — primary: ['sglang', 'vllm'], all: [...]
7. Librarian: Running 4 hybrid searches (2 sub-queries × 2 collections)
8. Librarian: Retrieved 11 unique documents total
9. Analyst: Pre-filter skipped 4 irrelevant doc(s) (primary terms: ['sglang', 'vllm'])
10. Analyst: 'SGLang: Efficient Execution of Structured ...' scored 9/10
11. Analyst: Confidence = 0.87 (3 high-quality docs from 7 total)
12. Writer: Ready to synthesize using local DB only (3 docs)
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agentic framework | LangGraph (StateGraph + MemorySaver checkpointing) |
| LLM | Ollama — `qwen2.5:7b-instruct` (local, configurable) |
| Embeddings | Ollama — `nomic-embed-text` (local) |
| Vector DB | ChromaDB (persistent, local) |
| Keyword search | `rank-bm25` (BM25Okapi) |
| Web search | DuckDuckGo via `ddgs` (privacy-preserving, no API key) |
| Full-text extraction | `trafilatura` |
| ArXiv ingestion | `arxiv` Python client |
| RSS ingestion | `feedparser` |
| UI | Streamlit |

---

## Configuration

All settings live in `src/config.py` and are overridable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `qwen2.5:7b-instruct` | Ollama model for reasoning and synthesis |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model for embeddings |
| `CHROMA_DB_PATH` | `./morningstar_db` | Local ChromaDB persistence path |
| `CONFIDENCE_THRESHOLD` | `0.6` | Below this, web fallback is triggered |
| `ENABLE_SMART_WEB_INGESTION` | `true` | Auto-learn from web results |
| `WEB_INGEST_MIN_SCORE` | `8` | Min LLM score to embed web content |
| `TIME_SENSITIVE_MAX_AGE_DAYS` | `1` | KB docs older than this are stale for live queries |
| `ARXIV_QUERY` | `cat:cs.AI OR cat:cs.CR` | Default ArXiv ingestion query |
| `ARXIV_MAX_RESULTS` | `20` | Papers fetched per ingestion run |
| `ARXIV_MIN_SCORE` | `7` | Min LLM score to embed ArXiv paper |

---

## Folder Structure (Current)

```
ai-labs/02-agents/agentic-morningstar/
├── README.md                         ← This document
│
├── langgraph-version/                ← IMPLEMENTED ✅
│   ├── src/
│   │   ├── main.py                   ← LangGraph graph definition + all node functions
│   │   ├── config.py                 ← All settings (env-overridable)
│   │   ├── graph/
│   │   │   ├── state.py              ← MorningstarState TypedDict + QueryIntent enum
│   │   │   └── __init__.py
│   │   ├── tools/
│   │   │   ├── chroma_tools.py       ← ChromaManager: hybrid search + RRF
│   │   │   ├── search_tools.py       ← DuckDuckGo + trafilatura extraction
│   │   │   ├── llm_tools.py          ← All Ollama interactions (score, decompose, synthesize, stream)
│   │   │   └── __init__.py
│   │   ├── ingestion/
│   │   │   ├── arxiv_fetcher.py      ← ArXiv daily ingestion pipeline
│   │   │   ├── web_ingestion.py      ← Smart web ingestion + single-topic manual ingest
│   │   │   ├── rss_fetcher.py        ← RSS/Atom feed subscription + ingestion
│   │   │   ├── backup.py             ← KB backup utilities
│   │   │   └── __init__.py
│   │   ├── agents/
│   │   │   └── __init__.py
│   │   └── utils/
│   │       └── logger.py
│   ├── app.py                        ← Streamlit UI (streaming synthesis, agent trace)
│   ├── rss_feeds.json                ← Persisted RSS subscriptions
│   ├── morningstar_db/               ← ChromaDB data (local, gitignored)
│   ├── requirements.txt
│   ├── .env                          ← LLM_MODEL, EMBEDDING_MODEL overrides
│   └── IMPROVEMENTS.md               ← Changelog and design decision log
│
└── crewai-version/                   ← PLANNED (Phase 2 comparison)
```

---

## Framework Comparison (Design Notes)

The LangGraph version was chosen first because it gives the most control. A CrewAI version is planned purely for comparison.

| Criterion | LangGraph | CrewAI | MS Agent Framework |
|-----------|:---------:|:------:|:-----------------:|
| Explicit state machine | ★★★ | ★★☆ | ★★☆ |
| Production / observability | ★★★ | ★★☆ | ★★★ |
| Code verbosity (less = better) | ★☆☆ | ★★★ | ★★☆ |
| Local / offline support | ★★★ | ★★★ | ★★☆ |
| Conditional routing control | ★★★ | ★★☆ | ★★☆ |
| Learning curve | ★★☆ | ★★★ | ★★☆ |

LangGraph wins on control and observability, which is why the agentic pipeline here uses explicit typed state (`MorningstarState`) and every edge is either deterministic or a named conditional — making the system debuggable and the `agent_reasoning` trace meaningful.

---

## Resources

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Ollama](https://ollama.ai/)
- [rank-bm25](https://github.com/dorianbrown/rank_bm25)
- [ddgs (DuckDuckGo Search)](https://github.com/deedy5/duckduckgo_search)
- [trafilatura](https://trafilatura.readthedocs.io/)

