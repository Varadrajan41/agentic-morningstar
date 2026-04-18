"""
LLM interaction tools using Ollama.
"""
import json
from typing import Dict, Any, Optional
import ollama

from src.config import LLM_MODEL


def chat_with_ollama(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = False,
    temperature: float = 0.7
) -> str:
    """
    Send a chat completion request to Ollama.
    
    Args:
        system_prompt: System instruction
        user_prompt: User message
        json_mode: Whether to request JSON output
        temperature: Sampling temperature
        
    Returns:
        LLM response text
    """
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]
    
    options = {'temperature': temperature}
    
    if json_mode:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            format='json',
            options=options
        )
    else:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            options=options
        )
    
    return response['message']['content'].strip()


def analyze_document_relevance(
    title: str,
    content: str,
    query: str
) -> Dict[str, Any]:
    """
    Analyze document relevance to a query.
    Returns score 1-10 and reasoning.
    
    This mirrors the scoring from digest_generator.py and web_Scout.py
    """
    system_prompt = """
    You are a strict Data Scientist evaluating research relevance.
    
    SCORING RUBRIC:
    - 1-3: Completely unrelated (e.g., color theory, physics, biology)
    - 4-7: General AI/ML, but not specifically relevant
    - 8-10: Highly relevant to query topic
    
    Output valid JSON with these exact keys:
    {
      "reasoning": "Explain your thought process in 1 sentence",
      "score": <integer 1-10>,
      "summary": "2-sentence summary of key points"
    }
    """
    
    user_prompt = f"""
Query: {query}

Document Title: {title}
Document Content: {content[:2000]}

Evaluate relevance and return JSON.
"""
    
    try:
        response = chat_with_ollama(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
            temperature=0.1
        )
        return json.loads(response)
    except json.JSONDecodeError:
        return {
            "reasoning": "Error parsing response",
            "score": 0,
            "summary": "Error in analysis"
        }


def synthesize_answer(
    query: str,
    documents: list,
    web_results: Optional[list] = None
) -> Dict[str, str]:
    """
    Synthesize final answer from retrieved documents.
    
    Args:
        query: Original user query
        documents: List of retrieved documents with text and metadata
        web_results: Optional web search results
        
    Returns:
        Dict with 'answer' and 'citations'
    """
    # Build context from documents
    context_parts = []
    citations = []
    
    for i, doc in enumerate(documents):
        meta = doc.get('metadata', {})
        title = meta.get('title', f'Document {i+1}')
        context_parts.append(f"Title: {title}\nContent: {doc['text']}\n")
        
        score_info = f" (Score: {meta.get('score', 'N/A')}/10)" if 'score' in meta else ""
        citations.append(f"[{title}]({doc.get('id', '#')}){score_info}")
    
    context = "\n\n".join(context_parts)
    
    # Add web results if available
    if web_results:
        web_context = "\n".join([
            f"Web Source: {r.get('title', 'Unknown')}\n{r.get('snippet', '')}"
            for r in web_results
        ])
        context += f"\n\nWeb Results:\n{web_context}"
        
        for r in web_results:
            citations.append(f"[Web: {r.get('title', 'Unknown')}]({r.get('url', '#')})")
    
    system_prompt = """
    You are Project Morningstar, a research assistant.
    Synthesize a clear, accurate answer using the provided context.
    Always cite your sources naturally in the text.
    If the context doesn't contain enough information, say so clearly.
    """
    
    user_prompt = f"""
Query: {query}

Context:
{context}

Provide a comprehensive answer with citations.
"""
    
    answer = chat_with_ollama(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7
    )
    
    return {
        'answer': answer,
        'citations': citations
    }


def rewrite_query_with_history(
    query: str,
    history: list,
    intent: str = "research"
) -> str:
    """
    Rewrite query into standalone search query using conversation history.
    
    This implements the query reformulation from app.py lines 95-107.
    
    Args:
        query: Latest user query
        history: List of previous messages [{role, content}, ...]
        intent: Query intent type
        
    Returns:
        Standalone search query
    """
    if not history:
        return query
    
    # Format history
    history_text = "\n".join([
        f"{m['role']}: {m['content']}"
        for m in history[-5:]  # Last 5 messages
    ])
    
    system_prompt = """
    Rewrite the user's latest query into a standalone search query.
    Use the conversation history for context.
    Output ONLY the rewritten query, nothing else.
    """
    
    user_prompt = f"""
Conversation History:
{history_text}

Latest Query: {query}

Rewrite as standalone search query:
"""
    
    rewritten = chat_with_ollama(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3
    )
    
    return rewritten if rewritten else query


def classify_query_intent(query: str) -> str:
    """
    Classify query intent for routing decisions.
    
    Returns one of: factual, research, comparison, summary, exploration
    """
    system_prompt = """
    Classify the query intent. Return ONLY one of:
    - factual: Simple fact lookup
    - research: Deep investigation needed
    - comparison: Comparing multiple items
    - summary: Request for summary/synthesis
    - exploration: Open-ended exploration
    """
    
    intent = chat_with_ollama(
        system_prompt=system_prompt,
        user_prompt=f"Query: {query}\n\nIntent:",
        temperature=0.1
    )
    
    valid_intents = ['factual', 'research', 'comparison', 'summary', 'exploration']
    intent_clean = intent.lower().strip()
    
    for valid in valid_intents:
        if valid in intent_clean:
            return valid
    
    return 'research'  # Default
