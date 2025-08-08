"""
Hybrid search implementation combining BM25 and vector search for improved retrieval.
"""

import bm25s
import Stemmer
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from loguru import logger
from langchain_chroma import Chroma
from langchain_core.documents import Document
from embeddings import get_embeddings_func
from get_config import config


class HybridSearcher:
    """Hybrid searcher combining BM25 and vector search."""

    def __init__(self, chroma_path: str = None, bm25_index_path: str = None):
        """
        Initialize hybrid searcher.

        Args:
            chroma_path: Path to ChromaDB database
            bm25_index_path: Path to BM25 index files
        """
        self.chroma_path = chroma_path or config.chroma_path
        self.bm25_index_path = Path(bm25_index_path or f"{self.chroma_path}_bm25")
        self.stemmer = Stemmer.Stemmer("english")

        # Initialize ChromaDB
        self.db = Chroma(
            persist_directory=self.chroma_path, embedding_function=get_embeddings_func()
        )

        # Load or initialize BM25 index
        self.bm25_index = None
        self.documents = []
        self.doc_metadata = []
        self.load_bm25_index()

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 indexing with stemming.

        Args:
            text: Input text

        Returns:
            List of stemmed tokens
        """
        # Simple tokenization (split on whitespace and punctuation)
        import re

        tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())

        # Apply stemming
        stemmed_tokens = self.stemmer.stemWords(tokens)
        return stemmed_tokens

    def build_bm25_index(self):
        """Build BM25 index from ChromaDB documents."""
        logger.info("Building BM25 index from ChromaDB...")

        # Get all documents from ChromaDB
        try:
            # Get all documents by searching with a very broad query
            all_results = self.db.similarity_search("medical health emergency", k=10000)

            if not all_results:
                logger.warning("No documents found in ChromaDB for BM25 indexing")
                return

            logger.info(f"Found {len(all_results)} documents for BM25 indexing")

            # Extract text and metadata
            self.documents = []
            self.doc_metadata = []

            for doc in all_results:
                self.documents.append(doc.page_content)
                self.doc_metadata.append(doc.metadata)

            # Preprocess documents
            logger.info("Preprocessing documents with stemming...")
            corpus_tokens = [self.preprocess_text(doc) for doc in self.documents]

            # Build BM25 index
            logger.info("Creating BM25 index...")
            self.bm25_index = bm25s.BM25()
            self.bm25_index.index(corpus_tokens)

            # Save index
            self.save_bm25_index()
            logger.info(
                f"BM25 index built and saved with {len(self.documents)} documents"
            )

        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            raise

    def save_bm25_index(self):
        """Save BM25 index and metadata to disk."""
        self.bm25_index_path.mkdir(parents=True, exist_ok=True)

        # Save BM25 index
        self.bm25_index.save(str(self.bm25_index_path / "bm25_index"))

        # Save documents and metadata
        with open(self.bm25_index_path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        with open(self.bm25_index_path / "metadata.pkl", "wb") as f:
            pickle.dump(self.doc_metadata, f)

    def load_bm25_index(self):
        """Load BM25 index and metadata from disk."""
        try:
            if not self.bm25_index_path.exists():
                logger.info("BM25 index not found, will build on first search")
                return

            # Load BM25 index
            self.bm25_index = bm25s.BM25.load(str(self.bm25_index_path / "bm25_index"))

            # Load documents and metadata
            with open(self.bm25_index_path / "documents.pkl", "rb") as f:
                self.documents = pickle.load(f)

            with open(self.bm25_index_path / "metadata.pkl", "rb") as f:
                self.doc_metadata = pickle.load(f)

            logger.info(f"Loaded BM25 index with {len(self.documents)} documents")

        except Exception as e:
            logger.warning(f"Could not load BM25 index: {e}. Will rebuild.")
            self.bm25_index = None

    def bm25_search(self, query: str, k: int = 10) -> List[Tuple[str, Dict, float]]:
        """
        Perform BM25 search.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (document_text, metadata, score) tuples
        """
        if self.bm25_index is None:
            self.build_bm25_index()

        if self.bm25_index is None or not self.documents:
            logger.warning("BM25 index is empty")
            return []

        # Preprocess query
        query_tokens = self.preprocess_text(query)

        if not query_tokens:
            return []

        # Search
        scores, indices = self.bm25_index.retrieve(
            query_tokens, k=min(k, len(self.documents))
        )

        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):  # bm25s returns nested arrays
            if idx < len(self.documents):
                results.append(
                    (self.documents[idx], self.doc_metadata[idx], float(score))
                )

        return results

    def hybrid_search(
        self, query: str, k: int = 5, bm25_weight: float = 0.3
    ) -> List[Tuple[Any, float]]:
        """
        Perform hybrid search combining BM25 and vector search.

        Args:
            query: Search query
            k: Number of results to return
            bm25_weight: Weight for BM25 scores (0-1), vector search gets (1-bm25_weight)

        Returns:
            List of (document, combined_score) tuples, sorted by score descending
        """
        # Get more results from each method to have more diversity
        search_k = k * 2

        # Vector search (existing method)
        vector_results = self.db.similarity_search_with_score(query, k=search_k)

        # BM25 search
        bm25_results = self.bm25_search(query, k=search_k)

        # Normalize and combine scores
        combined_results = {}

        # Process vector search results (similarity scores, higher = more similar)
        vector_scores = [
            1 - score for doc, score in vector_results
        ]  # Convert distance to similarity
        if vector_scores:
            max_vector_score = max(vector_scores)
            min_vector_score = min(vector_scores)
            vector_range = max_vector_score - min_vector_score or 1.0

            for (doc, _), raw_score in zip(vector_results, vector_scores):
                normalized_score = (raw_score - min_vector_score) / vector_range
                doc_key = doc.page_content[:100]  # Use first 100 chars as key
                combined_results[doc_key] = {
                    "doc": doc,
                    "vector_score": normalized_score,
                    "bm25_score": 0.0,
                }

        # Process BM25 results (BM25 scores, higher = more relevant)
        bm25_scores = [score for _, _, score in bm25_results]
        if bm25_scores:
            max_bm25_score = max(bm25_scores)
            min_bm25_score = min(bm25_scores)
            bm25_range = max_bm25_score - min_bm25_score or 1.0

            for doc_text, metadata, raw_score in bm25_results:
                normalized_score = (raw_score - min_bm25_score) / bm25_range
                doc_key = doc_text[:100]

                if doc_key in combined_results:
                    combined_results[doc_key]["bm25_score"] = normalized_score
                else:
                    # Create a document-like object for BM25-only results
                    doc = Document(page_content=doc_text, metadata=metadata)
                    combined_results[doc_key] = {
                        "doc": doc,
                        "vector_score": 0.0,
                        "bm25_score": normalized_score,
                    }

        # Calculate combined scores
        final_results = []
        for doc_key, data in combined_results.items():
            combined_score = (1 - bm25_weight) * data[
                "vector_score"
            ] + bm25_weight * data["bm25_score"]
            final_results.append((data["doc"], combined_score))

        # Sort by combined score (descending)
        final_results.sort(key=lambda x: x[1], reverse=True)

        # Return top k results
        return final_results[:k]

    def rebuild_index(self):
        """Force rebuild of BM25 index."""
        logger.info("Forcing BM25 index rebuild...")
        self.bm25_index = None
        self.build_bm25_index()


# Global hybrid searcher instance
_hybrid_searcher = None


def get_hybrid_searcher() -> HybridSearcher:
    """Get global hybrid searcher instance."""
    global _hybrid_searcher
    if _hybrid_searcher is None:
        _hybrid_searcher = HybridSearcher()
    return _hybrid_searcher


def hybrid_similarity_search_with_score(
    query: str, k: int = 5, bm25_weight: float = 0.3
) -> List[Tuple[Any, float]]:
    """
    Convenience function for hybrid search.

    Args:
        query: Search query
        k: Number of results to return
        bm25_weight: Weight for BM25 scores (0.3 = 30% BM25, 70% vector)

    Returns:
        List of (document, score) tuples
    """
    searcher = get_hybrid_searcher()
    return searcher.hybrid_search(query, k, bm25_weight)

