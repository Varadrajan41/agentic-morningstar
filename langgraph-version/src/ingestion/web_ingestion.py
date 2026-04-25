"""
Smart web content ingestion.
Integrates with query pipeline to auto-learn from web results.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
import ollama

from src.config import EMBEDDING_MODEL, COLLECTION_DEEP
from src.tools.chroma_tools import get_chroma_manager
from src.tools.llm_tools import analyze_document_relevance
from src.tools.search_tools import extract_full_text


def score_and_embed_web_content(
    title: str,
    url: str,
    snippet: str,
    query_context: str,
    min_score: int = 8,
    extract_full: bool = True
) -> Dict[str, Any]:
    """
    Score web content and embed if high-quality.
    
    Higher threshold (8/10) for web content since it's less curated than ArXiv.
    
    Args:
        title: Web page title
        url: URL
        snippet: Text snippet or summary
        query_context: What the user was searching for
        min_score: Minimum score to embed (default 8 for web content)
        extract_full: Whether to extract full page content
        
    Returns:
        Dict with embedding status
    """
    # Optionally extract full content
    content = snippet
    if extract_full:
        try:
            full_text = extract_full_text(url)
            if full_text:
                # Truncate to avoid embedding limits
                content = full_text[:4000]
        except Exception as e:
            print(f"⚠️ Could not extract full text from {url}: {e}")
    
    # Analyze relevance to the query context
    analysis = analyze_document_relevance(
        title=title,
        content=content,
        query=query_context
    )
    
    score = analysis.get("score", 0)
    
    result = {
        "title": title,
        "url": url,
        "score": score,
        "embedded": False,
        "collection": None
    }
    
    # Only embed high-quality web content (stricter than ArXiv)
    if score >= min_score:
        chroma = get_chroma_manager()
        
        # Prepare document text
        doc_text = f"Title: {title}\nURL: {url}\nContent: {content}\nAI Summary: {analysis.get('summary', '')}"
        
        # Generate embedding
        embedding_response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=doc_text
        )
        embedding = embedding_response['embedding']
        
        # Store in deep_dive collection (web content is "deep" research)
        date_str = datetime.now().strftime("%Y-%m-%d")
        doc_id = f"web_{url.replace('/', '_').replace(':', '_')[:100]}"
        
        chroma.collections[COLLECTION_DEEP].upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[{
                "title": title,
                "url": url,
                "date_ingested": date_str,
                "score": score,
                "type": "web_content",
                "query_context": query_context,
                "ai_summary": analysis.get("summary", "")
            }]
        )
        
        result["embedded"] = True
        result["collection"] = COLLECTION_DEEP
        result["doc_id"] = doc_id
        print(f"🌐 Embedded web content: {title[:60]}... (score: {score}/10)")
    else:
        print(f"🗑️ Rejected web content: {title[:60]}... (score: {score}/10)")
    
    return result


def ingest_web_results(
    web_results: List[Dict],
    query_context: str,
    min_score: int = 8
) -> Dict[str, Any]:
    """
    Ingest multiple web search results.
    
    Used by query pipeline when web fallback is triggered.
    Learns from high-quality web sources for future queries.
    
    Args:
        web_results: List of web search results
        query_context: Original user query
        min_score: Minimum score for embedding
        
    Returns:
        Ingestion stats
    """
    print(f"\n🌐 Smart Web Ingestion: Learning from {len(web_results)} results")
    print("=" * 50)
    
    embedded_count = 0
    rejected_count = 0
    results = []
    
    for result in web_results:
        try:
            ingest_result = score_and_embed_web_content(
                title=result.get("title", "Unknown"),
                url=result.get("url", ""),
                snippet=result.get("snippet", ""),
                query_context=query_context,
                min_score=min_score
            )
            
            results.append(ingest_result)
            
            if ingest_result["embedded"]:
                embedded_count += 1
            else:
                rejected_count += 1
                
        except Exception as e:
            print(f"⚠️ Error ingesting web result: {e}")
            rejected_count += 1
    
    print("\n" + "=" * 50)
    print(f"✅ Web Ingestion Complete!")
    print(f"   Processed: {len(web_results)}")
    print(f"   Embedded to knowledge base: {embedded_count}")
    print(f"   Rejected (low quality): {rejected_count}")
    
    return {
        "total": len(web_results),
        "embedded": embedded_count,
        "rejected": rejected_count,
        "results": results
    }


def ingest_single_topic(
    topic: str,
    max_results: int = 5,
    min_score: int = 8
) -> Dict[str, Any]:
    """
    Manual topic ingestion - search web and embed high-quality results.
    
    This is used for on-demand learning via Streamlit UI.
    
    Args:
        topic: Topic to learn about
        max_results: Max web results to fetch
        min_score: Minimum quality score
        
    Returns:
        Ingestion stats
    """
    from src.tools.search_tools import web_search_with_extraction
    
    print(f"\n📚 Learning about: {topic}")
    print("=" * 50)
    
    # Search web for topic
    web_results = web_search_with_extraction(
        query=topic,
        max_results=max_results,
        extract_full=False
    )
    
    if not web_results:
        print("⚠️ No web results found")
        return {"total": 0, "embedded": 0, "rejected": 0}
    
    # Ingest results
    return ingest_web_results(
        web_results=web_results,
        query_context=topic,
        min_score=min_score
    )


if __name__ == "__main__":
    # Test manual ingestion
    topic = "Agentic RAG latest advances 2026"
    stats = ingest_single_topic(topic)
    print(f"\nLearned {stats['embedded']} new sources about: {topic}")
