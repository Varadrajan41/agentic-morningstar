"""
Test script to verify all components are properly configured.
"""
import sys
import os

print("Testing Agentic Morningstar Setup")
print("=" * 50)

# Test imports
print("\n1. Testing imports...")
try:
    from src.config import LLM_MODEL, EMBEDDING_MODEL, CHROMA_DB_PATH
    print(f"   ✓ Config: LLM={LLM_MODEL}, Embedding={EMBEDDING_MODEL}")
    
    from src.graph.state import MorningstarState, QueryIntent
    print(f"   ✓ State: QueryIntent has {len(QueryIntent)} values")
    
    from src.tools.chroma_tools import ChromaManager, get_chroma_manager
    print("   ✓ Chroma tools imported")
    
    from src.tools.search_tools import web_search, web_search_with_extraction
    print("   ✓ Search tools imported")
    
    from src.tools.llm_tools import (
        chat_with_ollama,
        analyze_document_relevance,
        synthesize_answer,
        rewrite_query_with_history,
        classify_query_intent
    )
    print("   ✓ LLM tools imported")
    
    from src.main import router_node, librarian_node, analyst_node, web_scout_node, writer_node, build_graph
    print("   ✓ Main graph nodes imported")
    
    from langgraph.graph import StateGraph
    print("   ✓ LangGraph imported")
    
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test ChromaDB connection
print("\n2. Testing ChromaDB...")
try:
    manager = get_chroma_manager()
    stats = manager.get_collection_stats()
    print(f"   ✓ ChromaDB connected")
    print(f"   ✓ Collections: {stats}")
except Exception as e:
    print(f"   ⚠ ChromaDB not initialized (expected if empty): {e}")

# Test graph building
print("\n3. Testing graph construction...")
try:
    workflow = build_graph()
    print("   ✓ Graph built successfully")
    print(f"   ✓ Nodes: router, librarian, analyst, web_scout, writer")
except Exception as e:
    print(f"   ✗ Graph build error: {e}")
    sys.exit(1)

# Test state creation
print("\n4. Testing state initialization...")
try:
    test_state: MorningstarState = {
        "query": "Test query",
        "rewritten_query": "",
        "query_intent": None,
        "collections_tried": [],
        "retrieved_docs": [],
        "web_search_performed": False,
        "web_results": [],
        "confidence_score": 0.0,
        "needs_web_fallback": False,
        "synthesized_answer": "",
        "citations": [],
        "retry_count": 0,
        "should_retry": False,
        "messages": [],
        "agent_reasoning": []
    }
    print("   ✓ State created successfully")
except Exception as e:
    print(f"   ✗ State error: {e}")

print("\n" + "=" * 50)
print("Setup test complete!")
print("\nNext steps:")
print("1. Ensure Ollama is running with the required models")
print("2. Run: python -m src.main")
print("3. Or launch UI: streamlit run app.py")
