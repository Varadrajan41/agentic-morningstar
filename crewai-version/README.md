# Agentic Morningstar - CrewAI Version

A fully agentic research assistant built with CrewAI, using role-based agents for intuitive collaboration.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Research Crew                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │   Research   │   │   Research   │   │   Technical  │   │
│  │   Librarian  │──▶│   Analyst    │──▶│   Writer     │   │
│  │              │   │              │   │              │   │
│  └──────────────┘   └──────────────┘   └──────────────┘   │
│         │                   │                   │           │
│         └───────────────────┴───────────────────┘           │
│                         │                                  │
│                         ▼                                  │
│               ┌──────────────┐                            │
│               │   Web Scout  │ (fallback)                  │
│               │   (Agent)    │                            │
│               └──────────────┘                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Agents

### 1. Research Librarian
```python
Agent(
    role="Research Librarian",
    goal="Query ChromaDB collections for relevant documents",
    backstory="""
    You are an expert in semantic search and hybrid retrieval.
    You know when to use dense embeddings vs keyword search.
    You query multiple ChromaDB collections efficiently.
    """,
    tools=[ChromaSearchTool, BM25SearchTool, FusionTool]
)
```

### 2. Research Analyst
```python
Agent(
    role="Research Analyst",
    goal="Evaluate document relevance and synthesize insights",
    backstory="""
    You are a senior data scientist at Authmind.
    You specialize in Identity Security, RAG, and Agentic AI.
    You score documents 1-10 based on strict rubrics.
    """
)
```

### 3. Technical Writer
```python
Agent(
    role="Technical Writer",
    goal="Generate clear, sourced responses",
    backstory="""
    You excel at explaining complex technical concepts.
    You always cite your sources clearly.
    """
)
```

### 4. Web Scout
```python
Agent(
    role="Web Scout",
    goal="Find web sources when local data is insufficient",
    backstory="""
    You are an expert web researcher.
    You use DuckDuckGo for privacy-preserving search.
    """,
    tools=[DuckDuckGoTool, TrafilaturaTool]
)
```

## Crew Definition

```python
from crewai import Crew, Task, Process

morningstar_crew = Crew(
    agents=[librarian, analyst, writer, web_scout],
    tasks=[query_task, evaluate_task, write_task, fallback_task],
    process=Process.hierarchical,  # Manager orchestrates
    manager_llm="ollama/qwen2.5:7b-instruct",
    memory=True,  # Shared context between agents
    cache=True,
    planning=True  # Auto-plan complex queries
)
```

## Features

- ✅ **Role-Based Clarity**: Explicit agent roles with backstories
- ✅ **CrewAI Flows**: Event-driven orchestration with decorators
- ✅ **14x Less Code**: Compared to equivalent LangGraph implementation
- ✅ **Gradual Autonomy**: Start with 100% human review, reduce as system proves itself
- ✅ **Built-in RAG Tools**: CrewAI has RAG capabilities out of the box

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the crew
python -m src.main
```

## Comparison with LangGraph Version

| Aspect | CrewAI | LangGraph |
|--------|--------|-----------|
| Code Verbosity | Terse | Verbose |
| Learning Curve | Gentle | Steep |
| Control | High-level | Fine-grained |
| Observability | Good | Excellent |
| Production | Ready | Ready |

## Future Enhancements

- [ ] CrewAI Flows with `@start`, `@listen`, `@router`
- [ ] Multiple crews for different query types
- [ ] Human-in-the-loop with gradual autonomy
- [ ] Structured output with Pydantic models
