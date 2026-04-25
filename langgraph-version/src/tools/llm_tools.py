"""
LLM interaction tools using Ollama.
"""
import json
from typing import Dict, Any, List, Optional
import ollama

from src.config import LLM_MODEL

# Single client instance with a timeout so a hung Ollama never freezes the app.
# Timeout applies per HTTP request (prompt → first token); generation continues
# streaming after that, so 60s is generous for local models.
LLM_TIMEOUT = 60  # seconds
_ollama_client = ollama.Client(timeout=LLM_TIMEOUT)


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
        response = _ollama_client.chat(
            model=LLM_MODEL,
            messages=messages,
            format='json',
            options=options
        )
    else:
        response = _ollama_client.chat(
            model=LLM_MODEL,
            messages=messages,
            options=options
        )
    
    return response['message']['content'].strip()


def check_keyword_match(title: str, content: str, query: str) -> bool:
    """
    Check if query keywords appear in document title or content.
    Returns True if keywords found, False otherwise.
    """
    query_lower = query.lower()
    title_lower = title.lower()
    content_lower = content.lower()
    
    # Extract key terms from query (remove common words)
    common_words = {'what', 'is', 'are', 'the', 'a', 'an', 'in', 'of', 'for', 'to', 'how', 'why', 'when', 'where'}
    query_terms = [term for term in query_lower.split() if term not in common_words and len(term) > 2]
    
    # Check if any significant query term appears in title
    title_matches = sum(1 for term in query_terms if term in title_lower)
    
    # Check if query terms appear in content
    content_matches = sum(1 for term in query_terms if term in content_lower)
    
    # Return True if at least one term in title, or multiple in content
    return title_matches > 0 or content_matches >= len(query_terms) // 2


def generate_feed_context(feed_title: str, entry_titles: List[str]) -> str:
    """
    Auto-detect the scoring context for an RSS feed by asking the LLM to
    summarise its topic from a sample of entry titles.

    Args:
        feed_title:    The feed's channel title (e.g. "cs.AI - Artificial Intelligence")
        entry_titles:  Up to 8 recent article/entry titles from the feed

    Returns:
        A short phrase (< 20 words) describing the feed's topic, suitable for
        use as the `query_context` argument in `score_and_embed_web_content`.
        Falls back to feed_title on any error.
    """
    sample = "\n".join(f"- {t}" for t in entry_titles[:8])

    system_prompt = (
        "You are a topic classifier. Based on a feed title and a sample of recent article titles, "
        "write a SHORT topic description (max 20 words) that captures the feed's subject matter. "
        "Focus on the specific research areas, technologies, or themes. "
        "Return ONLY the description — no explanation, no quotes, no JSON."
    )
    user_prompt = f"Feed title: {feed_title}\n\nRecent articles:\n{sample}\n\nTopic description:"

    try:
        result = chat_with_ollama(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1
        ).strip().strip('"').strip("'")
        return result if result else feed_title
    except Exception:
        return feed_title


def decompose_query(query: str, intent: str) -> List[str]:
    """
    Split a complex multi-part query into 2-3 atomic sub-queries for parallel retrieval.

    Simple or single-concept queries are returned unchanged as a one-element list.
    Decomposition is only attempted for COMPARISON, RESEARCH, and EXPLORATION intents.

    Args:
        query: The (possibly rewritten) user query.
        intent: Query intent string ('comparison', 'research', 'exploration', etc.)

    Returns:
        List of 1-3 sub-query strings. Always has at least one element.
    """
    system_prompt = """You are a query decomposition specialist for a research retrieval system.

Analyze the user query and decide if it contains MULTIPLE distinct information needs.

DECOMPOSE if the query:
- Compares or contrasts two or more concepts
- Asks about different aspects of different topics in one sentence
- Uses connectors like "and also", "as well as", "vs", "versus", "compared to",
  "plus", "additionally", "both ... and"

DO NOT decompose if the query:
- Asks about a single concept (even if phrased elaborately)
- Is already focused on one topic
- Is a simple factual or definition question

CRITICAL RULES:
- Maximum 3 sub-queries — never more
- Each sub-query must be fully self-contained and independently searchable
- Sub-queries should collectively cover the original query without overlap
- Prefer 2 sub-queries over 3 unless there are clearly 3 distinct aspects
- Return ONLY valid JSON with no markdown fences

EXAMPLES:
Query: "Compare RAG vs Agentic RAG and list healthcare use cases"
→ {"sub_queries": ["What is RAG and how does retrieval augmented generation work?", "What is Agentic RAG and how does it differ from traditional RAG?", "Healthcare use cases for RAG systems"]}

Query: "Explain quantization techniques for LLMs"
→ {"sub_queries": ["Explain quantization techniques for LLMs"]}

Query: "What are the advantages of vLLM and how does SGLang compare to it?"
→ {"sub_queries": ["What are the advantages of vLLM for LLM inference?", "How does SGLang compare to vLLM?"]}
"""

    user_prompt = f"""Query: {query}
Intent: {intent}

Return JSON: {{"sub_queries": ["sub_query_1", ...]}}
If no decomposition is needed: {{"sub_queries": ["{query}"]}}"""

    try:
        response = chat_with_ollama(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
            temperature=0.1
        )
        data = json.loads(response)
        sub_queries = data.get("sub_queries", [])
        if not isinstance(sub_queries, list) or not sub_queries:
            return [query]
        # Sanitise: non-empty strings only, cap at 3
        clean = [str(q).strip() for q in sub_queries if str(q).strip()]
        return clean[:3] if clean else [query]
    except Exception:
        return [query]


def analyze_document_relevance(
    title: str,
    content: str,
    query: str
) -> Dict[str, Any]:
    """
    Analyze document relevance to a query with strict title-focused scoring.
    
    The TITLE is the primary indicator - if it doesn't show the document
    is primarily about the query topic, the score should be LOW even if
    the query is mentioned in passing in the content.
    
    Returns score 1-10 and reasoning.
    """
    # First check: keyword validation
    has_keyword_match = check_keyword_match(title, content, query)
    
    system_prompt = """
    You are a strict Data Scientist evaluating research relevance.
    
    CRITICAL SCORING RULES:
    1. The TITLE is the PRIMARY indicator of what a document is about
    2. If the TITLE is about a DIFFERENT concept than the query, score LOW (1-3)
    3. A query topic mentioned only in passing in content is NOT relevant
    4. Documents must be PRIMARILY about the query topic to score 8-10
    5. Never give high scores to documents with mismatched topics
    
    SCORING RUBRIC:
    - 1-3: Mismatched topic (title about different concept, or tangential mention only)
      * Examples: Query "RAG" → Title "Agent Harness" (different concepts)
      * Examples: Query "transformers" → Title "CNNs" (wrong architecture)
    - 4-5: Related domain but different focus
      * Examples: Query "RAG" → Title "LLMs" (same field, not specific topic)
      * Examples: Query "cybersecurity" → Title "AI safety" (related but not query)
    - 6-7: Partially relevant, mentions query but not primary focus
    - 8-10: Directly about the query topic (title clearly indicates this)
      * Examples: Query "RAG" → Title "What is RAG in LLMs"
      * Examples: Query "transformers" → Title "BERT: Pre-training Transformers"
    
    EXPLICIT EXAMPLES:
    - Query: "RAG" → Doc: "Agent Harness Engineering" → Score: 2/10 (wrong topic, title not about RAG)
    - Query: "RAG" → Doc: "Introduction to Python" → Score: 1/10 (completely unrelated)
    - Query: "RAG" → Doc: "What is RAG and How It Works" → Score: 10/10 (direct match)
    - Query: "transformers" → Doc: "CNNs for Image Classification" → Score: 1/10 (wrong architecture)
    - Query: "transformers" → Doc: "BERT Architecture" → Score: 9/10 (directly about transformers)
    - Query: "multi-agent" → Doc: "Single LLM Performance" → Score: 3/10 (different concept)
    
    Output valid JSON with these exact keys:
    {
      "reasoning": "Explain title match and why score was assigned",
      "score": <integer 1-10>,
      "summary": "2-sentence summary of what the document is actually about"
    }
    """
    
    user_prompt = f"""
Query: {query}

Document Title: {title}

Document Content (truncated):
{content[:2000]}

INSTRUCTION: Does the TITLE indicate this document is primarily ABOUT "{query}"?
If the title is clearly about a different concept, give a LOW score (1-3) even if the content mentions the query.

Return JSON.
"""
    
    try:
        response = chat_with_ollama(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
            temperature=0.1
        )
        result = json.loads(response)
        
        # Post-processing: validate with keyword check
        score = result.get("score", 0)
        
        # If no keyword match and score is high, reduce it
        if not has_keyword_match and score >= 7:
            result["score"] = min(score, 4)  # Cap at 4 if no keyword match
            result["reasoning"] = f"[AUTO-CORRECTED] {result.get('reasoning', '')} - Score reduced: query keywords not found in document."
        
        # If score is suspiciously high but no keyword match in title, flag it
        if score >= 8 and not check_keyword_match(title, "", query):
            result["score"] = 3  # Force low score
            result["reasoning"] = f"[AUTO-CORRECTED] Title mismatch: Document appears to be about different topic than query."
        
        return result
        
    except json.JSONDecodeError:
        return {
            "reasoning": "Error parsing response",
            "score": 0,
            "summary": "Error in analysis"
        }


def _build_synthesis_prompts(
    query: str,
    documents: list,
    web_results: Optional[list] = None
) -> tuple:
    """
    Build system prompt, user prompt, and citations for synthesis.
    Shared by both the blocking and streaming synthesis functions.

    Returns:
        (system_prompt, user_prompt, citations)
    """
    context_parts = []
    citations = []

    for i, doc in enumerate(documents):
        meta = doc.get('metadata', {})
        title = meta.get('title', f'Document {i+1}')
        context_parts.append(f"Title: {title}\nContent: {doc['text']}\n")
        score_info = f" (Score: {meta.get('score', 'N/A')}/10)" if 'score' in meta else ""
        citations.append(f"[{title}]({doc.get('id', '#')}){score_info}")

    context = "\n\n".join(context_parts)

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
    return system_prompt, user_prompt, citations


def synthesize_answer(
    query: str,
    documents: list,
    web_results: Optional[list] = None
) -> Dict[str, str]:
    """
    Synthesize final answer (blocking). Used by CLI / non-streaming paths.

    Returns:
        Dict with 'answer' and 'citations'
    """
    system_prompt, user_prompt, citations = _build_synthesis_prompts(query, documents, web_results)
    answer = chat_with_ollama(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.7)
    return {'answer': answer, 'citations': citations}


def synthesize_answer_stream(
    query: str,
    documents: list,
    web_results: Optional[list] = None
):
    """
    Synthesize final answer as a token stream (generator).

    Yields:
        str token chunks as they arrive from Ollama.
        The final item yielded is a special dict {"citations": [...]} so the
        caller can extract citations without a second call.

    Usage in Streamlit:
        full = ""
        citations = []
        for chunk in synthesize_answer_stream(query, docs, web):
            if isinstance(chunk, dict):
                citations = chunk["citations"]
            else:
                full += chunk
                placeholder.markdown(full + "▌")
        placeholder.markdown(full)
    """
    system_prompt, user_prompt, citations = _build_synthesis_prompts(query, documents, web_results)

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]

    stream = _ollama_client.chat(
        model=LLM_MODEL,
        messages=messages,
        options={'temperature': 0.7},
        stream=True
    )

    for chunk in stream:
        token = chunk['message']['content']
        if token:
            yield token

    # Signal end with citations dict
    yield {"citations": citations}


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


def generate_arxiv_query(research_topic: str, priority_keywords: str = "") -> str:
    """
    Convert natural language research topic to proper ArXiv query syntax.
    
    Uses LLM to understand the topic and generate appropriate ArXiv query
    with correct categories, boolean operators, and synonym expansion.
    
    Args:
        research_topic: Natural language description of research interest
        priority_keywords: Optional comma-separated keywords to prioritize
        
    Returns:
        Properly formatted ArXiv query string
    """
    system_prompt = """
    You are an expert at converting research interests into ArXiv search queries.

    ArXiv Query Syntax Rules:
    - cat:cs.AI = Artificial Intelligence papers
    - cat:cs.CR = Cryptography and Security
    - cat:cs.LG = Machine Learning
    - cat:cs.CL = Computation and Language (NLP)
    - cat:cs.IR = Information Retrieval
    - cat:cs.DB = Databases
    - cat:cs.SE = Software Engineering
    - Use AND sparingly — only between truly essential concepts. Too many ANDs = 0 results.
    - Use OR generously for synonyms/abbreviations/alternatives
    - Use quotes for multi-word phrases
    - Use parentheses for grouping OR alternatives

    CRITICAL RULE — Keep queries SIMPLE:
    - Maximum 2 AND clauses (category + core concept). More ANDs = fewer results.
    - Prefer broad OR groups over multiple ANDs.
    - For specific paper lookups (e.g. "AWQ paper"), use: abbreviation OR "full name"
    - NEVER chain more than 2 AND operators.

    Example conversions:
    - "AWQ paper" → cat:cs.LG AND ("AWQ" OR "activation-aware weight quantization" OR "weight quantization LLM")
    - "transformers in healthcare" → cat:cs.AI AND (transformer OR "attention mechanism") AND (healthcare OR medical OR clinical)
    - "multi-agent RAG systems" → cat:cs.AI AND ("multi-agent" OR multiagent) AND (RAG OR "retrieval augmented")
    - "LLM security" → (cat:cs.AI OR cat:cs.CR) AND (LLM OR "large language model") AND (security OR vulnerability OR attack)
    - "vector database performance" → (cat:cs.DB OR cat:cs.IR) AND ("vector database" OR "vector store") AND (performance OR scalability)

    Always include the most relevant cs category.
    """

    keyword_instruction = ""
    if priority_keywords.strip():
        keyword_instruction = f"\n\nPriority keywords to include: {priority_keywords}"

    user_prompt = f"""Research interest: {research_topic}{keyword_instruction}

Generate a SIMPLE ArXiv query that:
1. Uses at most ONE cs category filter (cat:cs.X)
2. Has at most 2 AND clauses total — prefer OR for synonyms
3. Wraps multi-word phrases in double quotes
4. Includes the abbreviation if the topic has one (e.g. AWQ, RAG, BERT)

Return ONLY the query string, nothing else. No explanation."""
    
    query = chat_with_ollama(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3
    )
    
    # Clean up the query - remove any markdown or extra text
    query = query.strip()
    if query.startswith('```') and query.endswith('```'):
        query = query[3:-3].strip()
    if query.startswith('`') and query.endswith('`'):
        query = query[1:-1].strip()
    
    # Ensure it has a category if missing
    if not query.startswith('cat:') and not query.startswith('all:'):
        query = f"cat:cs.AI AND ({query})"
    
    return query
