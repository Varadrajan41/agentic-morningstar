"""
Web search tools using DuckDuckGo.
"""
from typing import List, Dict
from ddgs import DDGS
import trafilatura

from src.config import WEB_SEARCH_MAX_RESULTS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def web_search(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> List[Dict]:
    """
    Perform web search using DuckDuckGo.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of search result dicts with title, url, snippet
    """
    try:
        with DDGS() as ddgs:
            # Quote only short (≤3 word) queries to avoid ambiguity
            # (e.g., "RAG" → "RAG" prevents ragtime music results).
            # Long sentences must NOT be quoted — exact-phrase matching on a full
            # sentence returns near-zero results on DuckDuckGo.
            words = query.split()
            search_query = f'"{query}"' if len(words) <= 3 else query
            results = ddgs.text(search_query, backend="html", max_results=max_results)
            
            return [
                {
                    "title": r.get('title', 'Unknown'),
                    "url": r.get('href', ''),
                    "snippet": r.get('body', '')
                }
                for r in results
            ]
    except Exception as e:
        logger.warning(f"Web search error: {e}")
        return []


def extract_full_text(url: str) -> str:
    """
    Extract full text content from a URL using trafilatura.
    
    Args:
        url: URL to extract content from
        
    Returns:
        Extracted text or empty string if failed
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded)
            return text if text else ""
    except Exception as e:
        logger.warning(f"Extraction error for {url}: {e}")
    
    return ""


def web_search_with_extraction(
    query: str,
    max_results: int = WEB_SEARCH_MAX_RESULTS,
    extract_full: bool = False
) -> List[Dict]:
    """
    Search web and optionally extract full text from results.
    
    Args:
        query: Search query
        max_results: Max search results
        extract_full: Whether to extract full page content
        
    Returns:
        List of results with title, url, snippet, and optionally full_text
    """
    results = web_search(query, max_results)
    
    if extract_full:
        for result in results:
            result['full_text'] = extract_full_text(result['url'])
    
    return results
