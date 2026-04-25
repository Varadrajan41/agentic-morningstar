"""
ArXiv paper fetching and ingestion.
Ported from digest_generator.py with agentic enhancements.
"""
import arxiv
from datetime import datetime
from typing import List, Dict, Any

from src.config import LLM_MODEL, EMBEDDING_MODEL, COLLECTION_DAILY
from src.tools.llm_tools import analyze_document_relevance
from src.tools.chroma_tools import get_chroma_manager, is_already_ingested
from src.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_arxiv_papers(
    query: str = "cat:cs.AI OR cat:cs.CR",
    max_results: int = 20,
    sort_by_relevance: bool = False
) -> List[arxiv.Result]:
    """
    Fetch papers from ArXiv.

    Args:
        query: ArXiv search query
        max_results: Maximum papers to fetch
        sort_by_relevance: If True, sort by relevance (best for specific paper
            lookup like "AWQ paper"). If False (default), sort by newest first
            (best for daily ingestion of recent research).

    Returns:
        List of arxiv.Result objects
    """
    sort_criterion = (
        arxiv.SortCriterion.Relevance if sort_by_relevance
        else arxiv.SortCriterion.SubmittedDate
    )
    sort_label = "relevance" if sort_by_relevance else "newest first"
    logger.info(f"🔍 Fetching top {max_results} papers from ArXiv (sorted by {sort_label})...")

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_criterion
    )
    
    return list(client.results(search))


def score_and_embed_paper(
    paper: arxiv.Result,
    min_score: int = 7,
    query_context: str = "Agentic AI, RAG, Identity Security, Cybersecurity, Multi-Agent Systems"
) -> Dict[str, Any]:
    """
    Score paper relevance and embed if high-quality.
    
    Args:
        paper: ArXiv paper result
        min_score: Minimum score to embed (1-10)
        query_context: The topic/interest to score relevance against.
            Pass the user's research topic so papers are scored in the right context.
        
    Returns:
        Dict with paper info and embedding status
    """
    # Skip if already in knowledge base (avoids redundant LLM + embedding calls)
    if is_already_ingested(paper.entry_id):
        return {
            "title": paper.title, "url": paper.entry_id,
            "score": 0, "embedded": False, "skipped": True
        }

    # Analyze with LLM using the actual user topic, not a hardcoded string
    analysis = analyze_document_relevance(
        title=paper.title,
        content=paper.summary,
        query=query_context
    )
    
    score = analysis.get("score", 0)
    
    result = {
        "title": paper.title,
        "url": paper.entry_id,
        "score": score,
        "summary": analysis.get("summary", ""),
        "reasoning": analysis.get("reasoning", ""),
        "embedded": False
    }
    
    # Only embed high-quality papers
    if score >= min_score:
        chroma = get_chroma_manager()
        
        # Prepare document text
        doc_text = f"Title: {paper.title}\nAbstract: {paper.summary}\nAI Summary: {analysis.get('summary', '')}"
        
        # Generate embedding via Ollama
        import ollama
        _client = ollama.Client(timeout=60)
        embedding_response = _client.embeddings(
            model=EMBEDDING_MODEL,
            prompt=doc_text
        )
        embedding = embedding_response['embedding']
        
        # Store in ChromaDB
        date_str = datetime.now().strftime("%Y-%m-%d")
        chroma.collections[COLLECTION_DAILY].upsert(
            ids=[paper.entry_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[{
                "title": paper.title,
                "date_ingested": date_str,
                "score": score,
                "type": "arxiv_paper",
                "url": paper.entry_id
            }]
        )
        
        result["embedded"] = True
        logger.info(f"⭐ Embedded: {paper.title[:60]}... (score: {score}/10)")
    else:
        logger.debug(f"🗑️ Rejected: {paper.title[:60]}... (score: {score}/10)")
    
    return result


def daily_arxiv_ingest(
    query: str = "cat:cs.AI OR cat:cs.CR",
    max_results: int = 20,
    min_score: int = 7,
    sort_by_relevance: bool = False,
    query_context: str = "",
    progress_callback=None
) -> Dict[str, Any]:
    """
    Daily ingestion workflow for ArXiv papers.

    Args:
        query: ArXiv search query
        max_results: Maximum papers to fetch
        min_score: Minimum quality score to embed
        sort_by_relevance: Sort by relevance (True) or newest first (False).
            Use True when searching for a specific paper by name/topic.
            Use False (default) for daily ingestion of recent research.
        progress_callback: Optional callable(current, total, title, embedded)

    Returns:
        Dict with ingestion stats
    """
    logger.info("🚀 Starting Daily ArXiv Ingestion")
    
    # Fetch papers
    papers = fetch_arxiv_papers(query, max_results, sort_by_relevance=sort_by_relevance)
    
    # Score and embed each
    embedded_count = 0
    rejected_count = 0
    results = []
    
    # Use provided context or fall back to the arxiv query itself as context
    scoring_context = query_context.strip() if query_context.strip() else query

    for i, paper in enumerate(papers):
        try:
            result = score_and_embed_paper(paper, min_score, query_context=scoring_context)
            results.append(result)
            
            if result["embedded"]:
                embedded_count += 1
            else:
                rejected_count += 1

            if progress_callback:
                progress_callback(i + 1, len(papers), paper.title, result.get("embedded", False))
                
        except Exception as e:
            logger.warning(f"⚠️ Error processing paper: {e}")
            rejected_count += 1
            if progress_callback:
                progress_callback(i + 1, len(papers), paper.title, False)
    
    logger.info(f"✅ Ingestion complete — fetched: {len(papers)}, embedded: {embedded_count}, rejected: {rejected_count}")
    
    return {
        "total_fetched": len(papers),
        "embedded": embedded_count,
        "rejected": rejected_count,
        "results": results
    }


if __name__ == "__main__":
    # Run daily ingestion
    stats = daily_arxiv_ingest()
    print(f"\nIngested {stats['embedded']} high-quality papers into knowledge base.")
