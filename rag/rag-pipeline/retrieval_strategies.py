"""Retrieval strategies for RAG pipeline - extracting core ideas from evaluation templates."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import bm25s
import numpy as np


class RetrievalStrategy(ABC):
    """Base class for retrieval strategies."""
    
    @abstractmethod
    def retrieve(self, query: str, document_store, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for the query."""
        pass


class DefaultRetrieval(RetrievalStrategy):
    """Default strategy: direct similarity search with the original query."""
    
    def retrieve(self, query: str, document_store, k: int = 5) -> List[Dict[str, Any]]:
        return document_store.search(query, k=k)


class HyDERetrieval(RetrievalStrategy):
    """HyDE strategy: Generate hypothetical answer first, then search."""
    
    def __init__(self, llm_client):
        self.llm = llm_client
        
    def retrieve(self, query: str, document_store, k: int = 5) -> List[Dict[str, Any]]:
        # Core HyDE idea: generate a hypothetical answer
        prompt = f"""You are an AI expert in emergency medicine and healthcare.
Generate a hypothetical but detailed answer to this medical question or statement.
Be specific and include medical terminology that would appear in relevant documents.

Question/Statement: {query}

Hypothetical Answer:"""
        
        try:
            response = self.llm.client.generate(
                model=self.llm.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_predict": 200
                }
            )
            hypothetical_answer = response["response"]
            
            # Use the hypothetical answer for retrieval
            return document_store.search(hypothetical_answer, k=k)
            
        except Exception as e:
            print(f"HyDE generation failed: {e}, falling back to default")
            return document_store.search(query, k=k)


class HybridRetrieval(RetrievalStrategy):
    """Hybrid strategy: Combine BM25 keyword search with semantic search."""
    
    def __init__(self, alpha: float = 0.5):
        """
        Initialize hybrid retrieval.
        
        Args:
            alpha: Weight for semantic search (1-alpha for BM25)
        """
        self.alpha = alpha
        self.bm25_index = None
        self.tokenized_corpus = None
        
    def _build_bm25_index(self, chunks: List[str]):
        """Build BM25 index from chunks."""
        # Tokenize chunks for BM25
        self.tokenized_corpus = bm25s.tokenize(chunks, stopwords="en")
        
        # Create and index
        self.bm25_index = bm25s.BM25()
        self.bm25_index.index(self.tokenized_corpus)
        
    def retrieve(self, query: str, document_store, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve using both BM25 and semantic search, then combine."""
        # Build BM25 index if not already built
        if self.bm25_index is None and hasattr(document_store, 'chunks'):
            print("Building BM25 index for hybrid search...")
            self._build_bm25_index(document_store.chunks)
        
        # Get semantic results
        semantic_results = document_store.search(query, k=k*2)  # Get more for merging
        
        # Get BM25 results
        bm25_results = []
        if self.bm25_index is not None:
            tokenized_query = bm25s.tokenize([query], stopwords="en")
            bm25_doc_ids, bm25_scores = self.bm25_index.retrieve(
                tokenized_query, k=k*2
            )
            
            # Convert BM25 results to same format
            for idx, score in zip(bm25_doc_ids[0], bm25_scores[0]):
                if idx < len(document_store.chunks):
                    bm25_results.append({
                        "chunk": document_store.chunks[idx],
                        "metadata": document_store.chunk_metadata[idx],
                        "score": float(score),
                        "source": "bm25"
                    })
        
        # Combine results with weighted scores
        combined_results = {}
        
        # Add semantic results
        for result in semantic_results:
            chunk = result["chunk"]
            combined_results[chunk] = {
                "chunk": chunk,
                "metadata": result["metadata"],
                "semantic_score": result["score"],
                "bm25_score": 0.0,
                "combined_score": self.alpha * result["score"]
            }
        
        # Add/update with BM25 results
        for result in bm25_results:
            chunk = result["chunk"]
            if chunk in combined_results:
                combined_results[chunk]["bm25_score"] = result["score"]
                # Normalize BM25 scores to [0,1] range like semantic scores
                normalized_bm25 = result["score"] / (1.0 + result["score"])
                combined_results[chunk]["combined_score"] += (1 - self.alpha) * normalized_bm25
            else:
                normalized_bm25 = result["score"] / (1.0 + result["score"])
                combined_results[chunk] = {
                    "chunk": chunk,
                    "metadata": result["metadata"],
                    "semantic_score": 0.0,
                    "bm25_score": result["score"],
                    "combined_score": (1 - self.alpha) * normalized_bm25
                }
        
        # Sort by combined score and return top k
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )[:k]
        
        # Format output
        return [
            {
                "chunk": r["chunk"],
                "metadata": r["metadata"],
                "score": r["combined_score"]
            }
            for r in sorted_results
        ]