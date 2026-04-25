# Ingestion Module

This module handles knowledge base creation and management for Agentic Morningstar.

## Features

### 1. ArXiv Daily Ingestion (`arxiv_fetcher.py`)
- Fetches latest AI/Security papers from ArXiv
- Scores papers 1-10 using LLM relevance rubric
- Only embeds high-quality papers (score ≥ 7)
- Stores in `daily_research` collection

### 2. Smart Web Ingestion (`web_ingestion.py`)
- Auto-learns from high-quality web search results
- Higher threshold (score ≥ 8) for web content
- Stores in `deep_dive_research` collection
- Triggered automatically when web fallback is used

### 3. Manual Topic Learning
- On-demand learning via Streamlit UI
- Search web for any topic and embed quality results
- Immediate knowledge base expansion

## Usage

### CLI

```bash
# Daily ArXiv ingestion
python -m src.ingestion arxiv

# Learn about specific topic
python -m src.ingestion web "Agentic RAG latest advances"

# Show knowledge base stats
python -m src.ingestion stats
```

### GitHub Actions (Automated)

Daily ingestion runs automatically via `.github/workflows/daily-ingestion.yml`:
- Scheduled: Every day at 6 AM UTC
- Manual trigger: Available in GitHub Actions UI
- Commits updated knowledge base back to repo

### Streamlit UI

In the sidebar:
- **🎓 Learn Now**: Enter topic, click to ingest immediately
- **📚 Run ArXiv Ingestion**: Manual trigger for daily fetch
- **🧠 Smart Web Learning**: Toggle auto-learning from web results

## How It Works

### Smart Ingestion Flow

```
User Query → Low Confidence → Web Search
                                ↓
                    High-Quality Results?
                           ↓
                Yes → Embed to ChromaDB
                 ↓
        Future queries benefit!
```

When the agent:
1. Can't find good local results (confidence < 0.6)
2. Searches web and finds relevant sources
3. Scores them 1-10 using same rubric as ArXiv
4. If score ≥ 8, automatically embeds to `deep_dive_research`
5. Next time same topic is queried → local hit!

## Configuration

Environment variables:
- `ENABLE_SMART_WEB_INGESTION`: Enable auto-learning (default: true)
- `WEB_INGEST_MIN_SCORE`: Quality threshold for web (default: 8)
- `ARXIV_QUERY`: ArXiv search query (default: "cat:cs.AI OR cat:cs.CR")
- `ARXIV_MAX_RESULTS`: Papers to fetch (default: 20)
- `ARXIV_MIN_SCORE`: Quality threshold for papers (default: 7)

## File Structure

```
src/ingestion/
├── __init__.py
├── __main__.py           # CLI entry point
├── arxiv_fetcher.py      # ArXiv daily ingestion
├── web_ingestion.py      # Smart web learning
└── README.md
```

## Comparison with Original

| Aspect | Original (digest_generator.py) | Agentic Version |
|--------|-------------------------------|-----------------|
| Sources | ArXiv only | ArXiv + Web |
| Trigger | Scheduled only | Scheduled + Smart + Manual |
| Selection | Top N papers | Quality-scored only |
| Auto-learn | No | Yes (from web results) |
| Scheduling | Windows Task Scheduler | GitHub Actions + Smart triggers |
