"""
ChromaDB and hybrid retrieval tools.
Implements vector search + BM25 + Reciprocal Rank Fusion.
"""
import os
from typing import List, Dict, Any, Tuple
import chromadb
from rank_bm25 import BM25Okapi
import ollama

from src.config import (
    CHROMA_DB_PATH, EMBEDDING_MODEL,
    COLLECTION_DAILY, COLLECTION_DEEP,
    VECTOR_K, BM25_K, FUSION_K, FINAL_TOP_K
)


class ChromaManager:
    """Manages ChromaDB connections and hybrid retrieval."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or CHROMA_DB_PATH
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Initialize collections
        self.collections = {
            "daily_research": self.client.get_or_create_collection(name=COLLECTION_DAILY),
            "deep_dive_research": self.client.get_or_create_collection(name=COLLECTION_DEEP)
        }
        
        # Cache for BM25 indices
        self._bm25_cache = {}
        self._data_cache = {}
        self._refresh_indices()
    
    def _refresh_indices(self):
        """Refresh BM25 indices for all collections."""
        for name, coll in self.collections.items():
            data = coll.get(include=['documents', 'metadatas'])
            self._data_cache[name] = data
            
            if data['documents']:
                tokenized = [doc.lower().split() for doc in data['documents']]
                self._bm25_cache[name] = BM25Okapi(tokenized)
            else:
                self._bm25_cache[name] = None
    
    def reciprocal_rank_fusion(
        self,
        dense_ranks: List[str],
        sparse_ranks: List[str],
        k: int = FUSION_K
    ) -> List[str]:
        """
        Combine vector and keyword search results using RRF.
        
        Args:
            dense_ranks: Document IDs from vector search (ordered by relevance)
            sparse_ranks: Document IDs from BM25 search (ordered by relevance)
            k: Constant for RRF calculation (default 60)
            
        Returns:
            Fused and ranked document IDs
        """
        rrf_scores = {}
        
        # Score from vector results
        for rank, doc_id in enumerate(dense_ranks):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
        # Score from BM25 results
        for rank, doc_id in enumerate(sparse_ranks):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        
        # Sort by score descending
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in sorted_docs]
    
    def hybrid_search(
        self,
        query: str,
        collection_name: str = "daily_research",
        vector_k: int = VECTOR_K,
        bm25_k: int = BM25_K,
        final_k: int = FINAL_TOP_K
    ) -> Tuple[List[str], List[str], List[Dict]]:
        """
        Perform hybrid search: vector + BM25 + fusion.
        
        Auto-refreshes indices before search to catch newly ingested content.
        
        Returns:
            Tuple of (document_ids, document_texts, metadata_list)
        """
        # Refresh indices to catch any newly ingested content
        self._refresh_indices()
        
        collection = self.collections.get(collection_name)
        if not collection:
            raise ValueError(f"Unknown collection: {collection_name}")
        
        # Get embedding for query
        embedding_response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=query)
        query_embedding = embedding_response['embedding']
        
        # Vector search
        vector_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=vector_k
        )
        vector_ids = vector_results['ids'][0] if vector_results['ids'] else []
        
        # BM25 search
        bm25 = self._bm25_cache.get(collection_name)
        bm25_ids = []
        
        if bm25:
            tokenized_query = query.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            data = self._data_cache[collection_name]
            
            # Get top BM25 results
            bm25_ranked = sorted(
                zip(data['ids'], bm25_scores),
                key=lambda x: x[1],
                reverse=True
            )
            bm25_ids = [doc_id for doc_id, score in bm25_ranked[:bm25_k] if score > 0]
        
        # Fuse results
        fused_ids = self.reciprocal_rank_fusion(vector_ids, bm25_ids)[:final_k]
        
        # Fetch full documents
        if fused_ids:
            results = collection.get(ids=fused_ids)
            return results['ids'], results['documents'], results['metadatas']
        
        return [], [], []
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get document counts for all collections."""
        stats = {}
        for name, coll in self.collections.items():
            data = coll.get()
            stats[name] = len(data['ids']) if data['ids'] else 0
        return stats


# Singleton instance
_chroma_manager = None

def get_chroma_manager() -> ChromaManager:
    """Get or create ChromaManager singleton."""
    global _chroma_manager
    if _chroma_manager is None:
        _chroma_manager = ChromaManager()
    return _chroma_manager
