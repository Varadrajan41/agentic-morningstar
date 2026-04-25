"""
ArXiv paper fetching and ingestion.
Ported from digest_generator.py with agentic enhancements.
"""
import arxiv
from datetime import datetime
from typing import List, Dict, Any

from src.config import LLM_MODEL, EMBEDDING_MODEL, COLLECTION_DAILY
from src.tools.llm_tools import analyze_document_relevance
from src.tools.chroma_tools import get_chroma_manager


def fetch_arxiv_papers(
    query: str = "cat:cs.AI OR cat:cs.CR",
    max_results: int = 20
) -> List[arxiv.Result]:
    """
    Fetch recent papers from ArXiv.
    
    Args:
        query: ArXiv search query (default: AI and Cryptography/Security)
        max_results: Maximum papers to fetch
        
    Returns:
        List of arxiv.Result objects
    """
    print(f"🔍 Fetching top {max_results} papers from ArXiv...")
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    return list(client.results(search))


def score_and_embed_paper(
    paper: arxiv.Result,
    min_score: int = 7
) -> Dict[str, Any]:
    """
    Score paper relevance and embed if high-quality.
    
    Args:
        paper: ArXiv paper result
        min_score: Minimum score to embed (1-10)
        
    Returns:
        Dict with paper info and embedding status
    """
    # Analyze with LLM
    analysis = analyze_document_relevance(
        title=paper.title,
        content=paper.summary,
        query="Agentic AI, RAG, Identity Security, Cybersecurity, Multi-Agent Systems"
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
        embedding_response = ollama.embeddings(
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
        print(f"⭐ Embedded: {paper.title[:60]}... (score: {score}/10)")
    else:
        print(f"🗑️ Rejected: {paper.title[:60]}... (score: {score}/10)")
    
    return result


def daily_arxiv_ingest(
    query: str = "cat:cs.AI OR cat:cs.CR",
    max_results: int = 20,
    min_score: int = 7
) -> Dict[str, Any]:
    """
    Daily ingestion workflow for ArXiv papers.
    
    This is the main entry point for scheduled ingestion.
    
    Returns:
        Dict with ingestion stats
    """
    print("🚀 Starting Daily ArXiv Ingestion")
    print("=" * 50)
    
    # Fetch papers
    papers = fetch_arxiv_papers(query, max_results)
    
    # Score and embed each
    embedded_count = 0
    rejected_count = 0
    results = []
    
    for paper in papers:
        try:
            result = score_and_embed_paper(paper, min_score)
            results.append(result)
            
            if result["embedded"]:
                embedded_count += 1
            else:
                rejected_count += 1
                
        except Exception as e:
            print(f"⚠️ Error processing paper: {e}")
            rejected_count += 1
    
    print("\n" + "=" * 50)
    print(f"✅ Ingestion Complete!")
    print(f"   Total fetched: {len(papers)}")
    print(f"   Embedded: {embedded_count}")
    print(f"   Rejected: {rejected_count}")
    
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
