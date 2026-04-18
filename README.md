# Project Morningstar: Agentic Harness Brainstorming

## Overview

This document explores rebuilding Project Morningstar (a local-first research assistant with Tiered RAG) using various **Agentic AI frameworks** to understand agentic harness design patterns.

**Current Status:** Project Morningstar exists in a private git repo and uses a hybrid RAG + partial LangGraph multi-agent approach.

**Goal:** Create new agentic implementations from scratch using different frameworks for comparison and learning.

---

## Current Architecture Analysis

### Existing Components (from baseline project)

| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | 180 | Streamlit UI with hybrid retrieval (vector + BM25 + fusion), dual collections, web fallback |
| `web_Scout.py` | 172 | LangGraph-based multi-agent pipeline (Scout → Analyst → Researcher → Librarian) |
| `digest_generator.py` | 184 | ArXiv ingestion with LLM scoring and ChromaDB storage |
| `query_morningstar.py` | - | CLI query tool |

### Current Design Patterns

1. **Tiered Memory**: Two ChromaDB collections
   - `daily_research`: Fast cards / summaries / snippets
   - `deep_dive_research`: Full technical content / papers

2. **Hybrid Retrieval**: Dense embeddings + BM25 keyword + Reciprocal Rank Fusion

3. **Multi-Agent (Partial)**: web_Scout.py uses LangGraph with:
   - Typed state (`AgentState`)
   - Explicit nodes: Scout → Analyst → Researcher → Librarian
   - Sequential edges

4. **Local-First Stack**:
   - Ollama (local LLM: qwen2.5:7b-instruct)
   - nomic-embed-text (local embeddings)
   - ChromaDB (persistent vector store)
   - DuckDuckGo (web search via ddgs)

---

## Framework Comparison

### 1. LangGraph (Production Control)

**Already partially used in web_Scout.py**

**Best for:** Explicit state machines, complex conditional routing, production deployments

**Pros:**
- Explicit `TypedDict` state with typed edges
- 91% task completion rate in benchmarks (2026)
- Checkpoint persistence (`MemorySaver`, `PostgresSaver`)
- Superior observability and debugging
- Conditional routing for complex flows

**Cons:**
- Verbose syntax for simple workflows
- Steeper learning curve

**Proposed Agentic Architecture:**

```
                    START
                      │
                ┌─────▼─────┐
                │   Router  │ ← Classifies query intent
                └─────┬─────┘
                      │
     ┌────────────────┼────────────────┐
     │                │                │
┌────▼────┐    ┌──────▼──────┐   ┌────▼────┐
│  RAG    │    │ Web Search  │   │ Hybrid  │
│  Node   │    │   Node      │   │  Node   │
└────┬────┘    └──────┬──────┘   └────┬────┘
     │                │               │
     └────────────────┼───────────────┘
                        │
                  ┌─────▼─────┐
                  │  Analyst  │ ← Evaluates result quality
                  └─────┬─────┘
                        │
              ┌─────────┴─────────┐
              │                   │
        ┌─────▼─────┐       ┌────▼────┐
        │Synthesize │       │ Self-   │
        │   Node    │       │ Correct │ ← Re-query if needed
        └─────┬─────┘       └────┬────┘
              │                   │
              └─────────┬─────────┘
                        │
                   ┌────▼────┐
                   │   END   │
                   └─────────┘
```

**State Definition:**
```python
class MorningstarState(TypedDict):
    query: str
    query_intent: str  # "factual", "research", "comparison"
    collections_tried: List[str]
    retrieved_docs: List[Document]
    web_results: List[SearchResult]
    confidence_score: float
    synthesized_answer: str
    citations: List[str]
    retry_count: int
```

---

### 2. CrewAI (Rapid Prototyping)

**Best for:** Role-based personas, readable code, rapid iteration

**Pros:**
- Simple role-based syntax
- 14x less code than equivalent LangGraph (DocuSign case study)
- 450M agents running monthly
- Built-in RAG tools and memory
- Flows with decorators: `@start`, `@listen`, `@router`

**Cons:**
- Less control for complex workflows
- Struggles with long-running workflows

**Proposed Agentic Architecture:**

```python
# agents/research_librarian.py
from crewai import Agent

research_librarian = Agent(
    role="Research Librarian",
    goal="Query vector databases for relevant documents",
    backstory="""
    You are an expert in semantic search and hybrid retrieval.
    You know when to use dense embeddings vs keyword search.
    You can query multiple ChromaDB collections efficiently.
    """,
    tools=[ChromaSearchTool, BM25SearchTool, FusionTool]
)

# agents/research_analyst.py
research_analyst = Agent(
    role="Research Analyst",
    goal="Evaluate and synthesize research findings",
    backstory="""
    You are a senior data scientist at Authmind.
    You specialize in Identity Security, RAG, and Agentic AI.
    You score documents 1-10 based on strict rubrics.
    """
)

# agents/web_scout.py
web_scout = Agent(
    role="Web Scout",
    goal="Find and extract web sources when local data is insufficient",
    backstory="""
    You are an expert web researcher.
    You use DuckDuckGo for privacy-preserving search.
    You can extract full page content with trafilatura.
    """,
    tools=[DuckDuckGoTool, TrafilaturaTool]
)

# agents/technical_writer.py
technical_writer = Agent(
    role="Technical Writer",
    goal="Generate clear, cited responses to user queries",
    backstory="""
    You excel at explaining complex technical concepts.
    You always cite your sources clearly.
    You adapt tone based on query type.
    """
)
```

**Crew Definition:**
```python
from crewai import Crew, Task, Process

morningstar_crew = Crew(
    agents=[
        research_librarian,
        research_analyst,
        web_scout,
        technical_writer
    ],
    tasks=[
        query_collections_task,
        evaluate_results_task,
        fallback_web_search_task,
        synthesize_response_task
    ],
    process=Process.hierarchical,  # Manager orchestrates
    manager_llm="ollama/qwen2.5:7b-instruct",
    memory=True,  # Shared context
    cache=True,
    planning=True  # Auto-plan complex queries
)
```

---

### 3. Microsoft Agent Framework 1.0 (Enterprise)

**Released April 2026 - Successor to AutoGen**

**Best for:** Enterprise integration, A2A/MCP protocols, cross-runtime support

**Pros:**
- Production-ready v1.0
- Combines AutoGen + Semantic Kernel
- A2A (Agent-to-Agent) protocol support
- MCP (Model Context Protocol)
- Multi-provider model support
- .NET and Python SDKs

**Cons:**
- Newer framework, smaller community
- Requires learning new patterns

**Proposed Agentic Architecture:**

```python
# microsoft_morningstar/agents.py
from microsoft.agents import Agent
from microsoft.agents.orchestration import GraphOrchestrator
from microsoft.agents.memory import ContextProvider

class LibrarianAgent(Agent):
    def __init__(self):
        self.skills = ["chroma_query", "bm25_search", "fusion_ranking"]
    
    async def on_message(self, context, message):
        query = message.content
        collections = context.get("collections")
        
        # Query logic here
        results = await self.query_collections(collections, query)
        return results

class EvaluatorAgent(Agent):
    async def on_message(self, context, message):
        results = message.content
        
        # Score results with LLM
        scored = await self.score_relevance(results)
        return scored

# Orchestrator setup
orchestrator = GraphOrchestrator()

orchestrator.add_node("librarian", LibrarianAgent())
orchestrator.add_node("evaluator", EvaluatorAgent())
orchestrator.add_node("writer", WriterAgent())
orchestrator.add_node("web_scout", WebScoutAgent())

# Define edges with conditions
orchestrator.add_edge("librarian", "evaluator")
orchestrator.add_conditional_edge(
    "evaluator",
    lambda state: "web_scout" if state["confidence"] < 0.7 else "writer",
    ["web_scout", "writer"]
)
orchestrator.add_edge("web_scout", "writer")
```

---

### 4. LangChain Deep Agents (Batteries-Included)

**From January Labs / LangChain**

**Best for:** Auto-planning, sub-agent delegation, filesystem operations

**Pros:**
- Built-in `TodoListMiddleware` for planning
- Sub-agent delegation
- Filesystem middleware for caching
- Context management
- Built on LangChain + LangGraph

**Cons:**
- Less mature than pure LangGraph
- Smaller ecosystem

**Proposed Agentic Architecture:**

```python
# deepagents_morningstar/main.py
from deep_agents import Agent, TodoListMiddleware, FilesystemMiddleware

morningstar_agent = Agent(
    name="MorningstarResearcher",
    system_prompt="""
    You are Project Morningstar, an agentic research assistant.
    
    Your capabilities:
    1. Query local ChromaDB collections (daily_research, deep_dive)
    2. Use hybrid retrieval (vector + BM25 + fusion)
    3. Fall back to web search if local results are insufficient
    4. Synthesize answers with proper citations
    
    For complex queries, break down into sub-tasks.
    Always cite your sources.
    """,
    middleware=[
        TodoListMiddleware,      # Auto-creates task lists
        FilesystemMiddleware     # Cache results to disk
    ],
    tools=[
        ChromaQueryTool,
        BM25SearchTool,
        FusionRankTool,
        DuckDuckGoTool,
        TrafilaturaTool
    ],
    llm="ollama/qwen2.5:7b-instruct"
)

# Example: Complex query handling
result = await morningstar_agent.run(
    "Compare the latest Agentic RAG approaches from 2026 "
    "and identify which would work best for cybersecurity research"
)

# TodoList automatically creates:
# 1. [ ] Query daily_research for "Agentic RAG 2026"
# 2. [ ] Query deep_dive for technical papers
# 3. [ ] Evaluate result quality
# 4. [ ] If insufficient, search web for "Agentic RAG 2026"
# 5. [ ] Analyze and compare approaches
# 6. [ ] Filter for cybersecurity applicability
# 7. [ ] Synthesize final comparison
```

---

## Decision Matrix

| Criterion | LangGraph | CrewAI | MS Agent Framework | Deep Agents |
|-----------|:---------:|:------:|:------------------:|:-----------:|
| **Current Fit** | ★★★ | ★★☆ | ★☆☆ | ★★☆ |
| **Learning Curve** | ★★☆ | ★★★ | ★★☆ | ★★★ |
| **Production Ready** | ★★★ | ★★★ | ★★★ | ★★☆ |
| **Observability** | ★★★ | ★★☆ | ★★★ | ★★☆ |
| **Code Verbosity** | ★☆☆ | ★★★ | ★★☆ | ★★☆ |
| **Local/Offline** | ★★★ | ★★★ | ★★☆ | ★★★ |
| **Enterprise** | ★★★ | ★★☆ | ★★★ | ★★☆ |
| **Documentation** | ★★★ | ★★★ | ★★☆ | ★★☆ |

---

## Recommended Implementation Strategy

### Phase 1: Foundation (LangGraph)
- Extend existing web_Scout.py pattern
- Create full agentic RAG pipeline
- Add checkpointing with `MemorySaver`
- Implement human-in-the-loop with `interrupt_before`

### Phase 2: Comparison Study (CrewAI)
- Rebuild identical functionality
- Compare code size, maintainability, ergonomics
- Document trade-offs

### Phase 3: Advanced Features (Optional)
- Deep Agents for auto-planning
- Or MS Agent Framework for A2A protocol

---

## Folder Structure

```
ai-labs/02-agents/agentic-morningstar/
├── README.md                    # This document
├── langgraph-version/           # Phase 1
│   ├── src/
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── librarian.py
│   │   │   ├── analyst.py
│   │   │   ├── web_scout.py
│   │   │   └── writer.py
│   │   ├── graph/
│   │   │   ├── __init__.py
│   │   │   ├── state.py
│   │   │   ├── nodes.py
│   │   │   └── edges.py
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── chroma_tools.py
│   │   │   ├── search_tools.py
│   │   │   └── llm_tools.py
│   │   ├── config.py
│   │   └── main.py
│   ├── requirements.txt
│   └── README.md
│
├── crewai-version/              # Phase 2
│   ├── src/
│   │   ├── agents/
│   │   ├── crews/
│   │   ├── tools/
│   │   └── main.py
│   ├── requirements.txt
│   └── README.md
│
└── comparison-report.md         # After both implementations
```

---

## Key Insights

1. **Project Morningstar is already halfway to agentic**
   - web_Scout.py uses LangGraph for multi-agent web research
   - Pattern: Scout → Analyst → Researcher → Librarian

2. **The RAG part (app.py) is NOT yet agentic**
   - Static pipeline with hybrid retrieval
   - No autonomous decision-making
   - No self-correction capability

3. **The Opportunity: True Agentic Morningstar**
   - System autonomously decides which collections to query
   - Self-corrects if results are poor
   - Can chain multiple searches
   - Maintains research state across sessions
   - Human-in-the-loop for critical decisions

4. **LangGraph is the natural next step**
   - web_Scout.py already uses it
   - Extend pattern to full RAG system

5. **CrewAI offers the best teaching moment**
   - Role-based syntax makes responsibilities explicit
   - "Research Librarian", "Analyst", "Writer", "Scout"

---

## Next Steps

1. **Initialize LangGraph version** in `langgraph-version/`
   - Set up project structure
   - Create requirements.txt with: langgraph, langchain, chromadb, ollama, streamlit, rank-bm25, ddgs, trafilatura
   
2. **Build core agents:**
   - Librarian: Hybrid retrieval (vector + BM25 + fusion)
   - Analyst: LLM-based relevance scoring
   - Router: Query intent classification
   - WebScout: Fallback web search
   - Writer: Response synthesis with citations

3. **Define LangGraph state and flow:**
   - Create `MorningstarState` TypedDict
   - Build state machine with conditional edges
   - Add checkpointing for persistence

4. **Create Streamlit UI:**
   - Integrate agentic graph
   - Show agent reasoning steps
   - Human-in-the-loop controls

5. **Then repeat with CrewAI** for comparison

---

## Resources

### Documentation
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [CrewAI Docs](https://docs.crewai.com/)
- [Microsoft Agent Framework](https://github.com/microsoft/agent-sdk)
- [Deep Agents Docs](https://deepagents.com/)

### Research Papers
- "DeepAgent: A General Reasoning Agent with Scalable Toolsets" (ArXiv 2025)
- "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation"

### Related Projects
- Original Project Morningstar (private git - for reference)
- Local Multilingual Assistant (`ai-labs/05-labs/`)

