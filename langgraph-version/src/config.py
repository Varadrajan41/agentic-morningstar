"""
Configuration settings for Agentic Morningstar.
"""
import os

# LLM Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# ChromaDB Configuration
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./morningstar_db")
COLLECTION_DAILY = "daily_research"
COLLECTION_DEEP = "deep_dive_research"

# Retrieval Configuration
VECTOR_K = 5
BM25_K = 5
FUSION_K = 60
FINAL_TOP_K = 6  # More candidates for analyst to score; analyst filters to high-quality ones

# Web Search Configuration
WEB_SEARCH_MAX_RESULTS = 5
WEB_FALLBACK_THRESHOLD = 0.7

# Agentic Configuration
MAX_RETRIES = 2
CONFIDENCE_THRESHOLD = 0.6

# Smart Ingestion Configuration
ENABLE_SMART_WEB_INGESTION = os.getenv("ENABLE_SMART_WEB_INGESTION", "true").lower() == "true"
WEB_INGEST_MIN_SCORE = 8  # Higher threshold for web content vs ArXiv (7)

# Staleness threshold for time-sensitive queries (e.g. live scores, prices, news).
# KB docs older than this many days will be considered stale and trigger a fresh
# web search even if they score well.  Configurable via env var.
TIME_SENSITIVE_MAX_AGE_DAYS = int(os.getenv("TIME_SENSITIVE_MAX_AGE_DAYS", "1"))
ARXIV_QUERY = os.getenv("ARXIV_QUERY", "cat:cs.AI OR cat:cs.CR")
ARXIV_MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "20"))
ARXIV_MIN_SCORE = int(os.getenv("ARXIV_MIN_SCORE", "7"))
