"""
CLI entry point for ingestion module.

Usage:
    python -m src.ingestion arxiv          # Run ArXiv daily ingestion
    python -m src.ingestion web "topic"    # Learn about specific topic
    python -m src.ingestion stats          # Show knowledge base stats
"""
import sys
from src.ingestion.arxiv_fetcher import daily_arxiv_ingest
from src.ingestion.web_ingestion import ingest_single_topic
from src.tools.chroma_tools import get_chroma_manager


def show_stats():
    """Display knowledge base statistics."""
    chroma = get_chroma_manager()
    stats = chroma.get_collection_stats()
    
    print("\n📊 Knowledge Base Statistics")
    print("=" * 50)
    for name, count in stats.items():
        print(f"  {name}: {count} documents")
    print("=" * 50)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.ingestion arxiv          # Daily ArXiv ingestion")
        print("  python -m src.ingestion web 'topic'    # Learn about topic")
        print("  python -m src.ingestion stats          # Show stats")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "arxiv":
        print("🚀 Running daily ArXiv ingestion...")
        stats = daily_arxiv_ingest()
        print(f"\n✅ Ingested {stats['embedded']} papers")
        
    elif command == "web":
        if len(sys.argv) < 3:
            print("Error: Please provide a topic to learn about")
            print("Example: python -m src.ingestion web 'Agentic RAG'")
            sys.exit(1)
        
        topic = sys.argv[2]
        print(f"🌐 Learning about: {topic}")
        stats = ingest_single_topic(topic)
        print(f"\n✅ Learned {stats['embedded']} new sources")
        
    elif command == "stats":
        show_stats()
        
    else:
        print(f"Unknown command: {command}")
        print("Available commands: arxiv, web, stats")
        sys.exit(1)


if __name__ == "__main__":
    main()
