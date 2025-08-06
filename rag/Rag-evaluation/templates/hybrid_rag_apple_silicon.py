"""
Hybrid RAG template for Apple Silicon - fast version using optimized data sources
Uses global BM25 and BioBERT FAISS for speed while leveraging pre-built indexes
"""

import json
import pickle
import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Fix OpenMP library conflicts on ARM Mac
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import faiss
import Stemmer
from sentence_transformers import SentenceTransformer

# Configure FAISS for ARM Mac stability
try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass

from llm_client import LocalLLMClient


class HybridRAGAppleSilicon:
    """Fast Hybrid RAG system using optimized data sources with minimal processing overhead."""

    def __init__(self, llm_client: LocalLLMClient = None):
        self.llm_client = llm_client or LocalLLMClient()
        self.stemmer = Stemmer.Stemmer("english")

        # Core optimized indexes only
        self.documents = []
        self.document_texts = []
        self.biobert_faiss_index = None
        self.global_bm25_index = None
        self.biobert_model = None

        # Load optimized indexes quickly
        self._load_optimized_indexes()

        print("✅ HybridRAGAppleSilicon fast version ready")

    def _load_optimized_indexes(self):
        """Load only essential optimized indexes for fast performance."""
        optimized_dir = Path(__file__).parent.parent / "optimized_indexes"

        if not optimized_dir.exists():
            raise FileNotFoundError(
                f"Optimized indexes directory not found: {optimized_dir}"
            )

        print(f"🚀 Loading essential indexes from {optimized_dir}...")

        # Load BioBERT FAISS index
        biobert_faiss_path = optimized_dir / "biobert_faiss.index"
        if biobert_faiss_path.exists():
            self.biobert_faiss_index = faiss.read_index(str(biobert_faiss_path))
            print(
                f"   ✅ Loaded BioBERT FAISS index ({biobert_faiss_path.stat().st_size / 1024 / 1024:.1f}MB)"
            )

        # Load global BM25 index
        bm25_path = optimized_dir / "bm25_index.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                self.global_bm25_index = pickle.load(f)
            print(
                f"   ✅ Loaded global BM25 index ({bm25_path.stat().st_size / 1024 / 1024:.1f}MB)"
            )

        # Load document mapping
        mapping_path = optimized_dir / "document_mapping.json"
        if mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            self.document_texts = [doc["text"] for doc in self.documents]
            print(
                f"   ✅ Loaded document mapping ({len(self.documents)} docs, {mapping_path.stat().st_size / 1024 / 1024:.1f}MB)"
            )

        # Load BioBERT model only if FAISS index is available
        if self.biobert_faiss_index is not None:
            print("🧠 Loading BioBERT model...")
            self.biobert_model = SentenceTransformer(
                "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
                device="mps" if hasattr(faiss, "StandardGpuResources") else "cpu",
            )

    def _tokenize_and_stem(self, text: str) -> List[str]:
        """Tokenize and stem text for BM25 processing."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        return self.stemmer.stemWords(tokens)

    def _safe_faiss_search(
        self, query_embeddings: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Safely perform FAISS search with proper data validation."""
        try:
            if not isinstance(query_embeddings, np.ndarray):
                query_embeddings = np.array(query_embeddings)

            # Convert to float32 and ensure C-contiguous memory layout
            query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)

            if query_embeddings.ndim == 1:
                query_embeddings = query_embeddings.reshape(1, -1)

            expected_dim = self.biobert_faiss_index.d
            if query_embeddings.shape[1] != expected_dim:
                raise ValueError(
                    f"Query embedding dimension {query_embeddings.shape[1]} doesn't match index dimension {expected_dim}"
                )

            distances, indices = self.biobert_faiss_index.search(query_embeddings, k)
            return distances, indices

        except Exception as e:
            print(f"⚠️  FAISS search failed: {e}")
            return np.array([[]]), np.array([[]])

    def _fast_hybrid_retrieval(self, question: str, k: int = 10) -> List[Dict]:
        """Fast hybrid retrieval using BioBERT FAISS + Global BM25 with simple fusion."""
        candidate_indices = set()

        # BioBERT semantic search
        if self.biobert_faiss_index is not None and self.biobert_model is not None:
            try:
                query_embedding = self.biobert_model.encode([question])
                _, faiss_indices = self._safe_faiss_search(query_embedding, k)
                candidate_indices.update(faiss_indices[0])
            except Exception as e:
                print(f"⚠️  BioBERT search failed: {e}")

        # Global BM25 keyword search
        if self.global_bm25_index is not None:
            try:
                query_tokens = self._tokenize_and_stem(question)
                bm25_results = self.global_bm25_index.retrieve([query_tokens], k=k)
                candidate_indices.update(bm25_results.documents[0])
            except Exception as e:
                print(f"⚠️  BM25 search failed: {e}")

        # Return candidate documents (no complex fusion)
        candidate_docs = [
            self.documents[i] for i in candidate_indices if i < len(self.documents)
        ]
        return candidate_docs[:k]

    def run(
        self, question: str, reference_contexts: List[str] = None
    ) -> Dict[str, Any]:
        """
        Process a medical statement using fast hybrid retrieval with optimized data sources.

        Pipeline:
        1. Fast Hybrid Retrieval (BioBERT FAISS + Global BM25)
        2. Direct LLM Classification

        Args:
            question: The medical statement to evaluate
            reference_contexts: Optional reference contexts (unused)

        Returns:
            Dict with 'answer' and 'context' keys
        """
        try:
            # Fast hybrid retrieval using optimized indexes
            candidate_docs = self._fast_hybrid_retrieval(question, k=5)

            # Prepare context and get LLM classification
            context = "\n\n".join(
                [
                    f"[{doc.get('article_title', 'Medical Text')}]: {doc['text']}"
                    for doc in candidate_docs
                ]
            )

            # Use LLM client for classification (no topic override)
            statement_is_true, statement_topic = self.llm_client.classify_statement(
                question, context
            )

            answer = {
                "statement_is_true": statement_is_true,
                "statement_topic": statement_topic,
            }

            retrieved_contexts = [doc["text"] for doc in candidate_docs]

            return {"answer": json.dumps(answer), "context": retrieved_contexts}

        except Exception as e:
            print(f"Error in fast HybridRAGAppleSilicon.run: {e}")
            import traceback

            traceback.print_exc()

            # Return safe defaults
            return {
                "answer": json.dumps({"statement_is_true": 1, "statement_topic": 0}),
                "context": [],
            }
