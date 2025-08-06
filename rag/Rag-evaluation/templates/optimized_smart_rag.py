import os
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any

# Fix OpenMP library conflicts on ARM Mac
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import bm25s
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import Stemmer

# Configure FAISS for ARM Mac stability
try:
    faiss.omp_set_num_threads(1)  # Prevent OpenMP threading issues
except Exception:
    pass

# Fallback for running script directly
try:
    from ..llm_client import LocalLLMClient
except (ImportError, ValueError):
    import sys

    sys.path.append(str(Path(__file__).parent.parent))
    from llm_client import LocalLLMClient


class OptimizedSmartRAG:
    """
    An optimized RAG template with pre-computed topic-specific BM25 indexes:
    1. Parallel Augmented Retrieval (BM25 + Semantic)
    2. Topic Selection via Majority Vote
    3. Focused BM25 Reranking using pre-computed topic indexes
    """

    def __init__(self, llm_client: LocalLLMClient = None):
        self.llm_client = llm_client or LocalLLMClient()
        self.stemmer = Stemmer.Stemmer("english")
        self.documents = []
        self.document_texts = []
        self.bm25_index = None
        self.faiss_index = None
        self.topic_bm25_indexes = {}  # Cache for topic-specific indexes
        self.topic_docs_mapping = {}  # Mapping from topic to documents
        self._load_optimized_indexes()
        # Load the model only once
        self.embedding_model = SentenceTransformer(
            "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        )

    def _load_optimized_indexes(self):
        """Loads pre-computed, optimized FAISS and BM25 indexes."""
        optimized_dir = Path(__file__).parent.parent / "optimized_indexes"
        faiss_path = optimized_dir / "biobert_faiss.index"
        bm25_path = optimized_dir / "bm25_index.pkl"
        mapping_path = optimized_dir / "document_mapping.json"

        if not all(p.exists() for p in [faiss_path, bm25_path, mapping_path]):
            raise FileNotFoundError(
                f"Optimized indexes not found in {optimized_dir}. "
                f"Please run `create_optimized_indexes.py` first."
            )

        print(f"🚀 Loading optimized indexes from {optimized_dir}...")
        self.faiss_index = faiss.read_index(str(faiss_path))
        with open(bm25_path, "rb") as f:
            self.bm25_index = pickle.load(f)
        with open(mapping_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        self.document_texts = [doc["text"] for doc in self.documents]

        # Load topic-specific indexes if available
        self._load_topic_specific_indexes()

        print(f"✅ Loaded {len(self.documents)} documents and indexes.")

    def _load_topic_specific_indexes(self):
        """Load pre-computed topic-specific BM25 indexes."""
        topic_dir = Path(__file__).parent.parent / "optimized_indexes" / "topic_bm25"
        mapping_file = topic_dir / "topic_index_mapping.json"

        if not mapping_file.exists():
            print("⚠️ Topic-specific indexes not found. Will create them dynamically.")
            return

        print("📚 Loading topic-specific BM25 indexes...")
        with open(mapping_file, "r", encoding="utf-8") as f:
            topic_mapping = json.load(f)

        for topic_name, info in topic_mapping.items():
            # Load the BM25 index
            index_path = topic_dir / info["index_file"]
            mapping_path = topic_dir / info["mapping_file"]

            if index_path.exists() and mapping_path.exists():
                with open(index_path, "rb") as f:
                    self.topic_bm25_indexes[topic_name] = pickle.load(f)

                with open(mapping_path, "r", encoding="utf-8") as f:
                    self.topic_docs_mapping[topic_name] = json.load(f)

        print(f"✅ Loaded {len(self.topic_bm25_indexes)} topic-specific BM25 indexes.")

    def _tokenize_and_stem(self, text: str) -> List[str]:
        """Basic tokenizer and stemmer for BM25."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        return self.stemmer.stemWords(tokens)

    def _generate_keyword_queries(self, query: str) -> List[str]:
        """Generates fact-focused keyword queries."""
        variations = [query]
        important_words = [
            w
            for w in query.split()
            if len(w) > 4 and w.lower() not in ["patient", "study"]
        ]
        if important_words:
            variations.append(" ".join(important_words))
        return variations

    def _augment_semantic_statements(self, query: str) -> List[str]:
        """Generates paraphrased and contradictory statements for semantic search."""
        variations = [query]
        if "higher" in query:
            variations.append(query.replace("higher", "lower"))
        if "increase" in query:
            variations.append(query.replace("increase", "decrease"))
        if "indicated for" in query:
            variations.append(query.replace("indicated for", "not recommended for"))
        return variations

    def _safe_faiss_search(self, query_embeddings: np.ndarray, k: int) -> tuple:
        """Safely perform FAISS search with proper data validation."""
        try:
            # Ensure proper data type and memory layout
            if not isinstance(query_embeddings, np.ndarray):
                query_embeddings = np.array(query_embeddings)

            # Convert to float32 and ensure C-contiguous memory layout
            query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)

            # Validate dimensions
            if query_embeddings.ndim == 1:
                query_embeddings = query_embeddings.reshape(1, -1)

            expected_dim = self.faiss_index.d
            if query_embeddings.shape[1] != expected_dim:
                raise ValueError(
                    f"Query embedding dimension {query_embeddings.shape[1]} doesn't match index dimension {expected_dim}"
                )

            # Perform the search
            distances, indices = self.faiss_index.search(query_embeddings, k)
            return distances, indices

        except Exception as e:
            print(f"⚠️ FAISS search failed: {e}")
            print(
                f"Query embeddings shape: {query_embeddings.shape}, dtype: {query_embeddings.dtype}"
            )
            print(f"Index dimension: {self.faiss_index.d}")
            # Return empty results as fallback
            return np.array([[]]), np.array([[]])

    def _get_topic_specific_results(
        self, winning_topic: str, question: str, k: int = 5
    ) -> List[Dict]:
        """Get results using pre-computed topic-specific BM25 index or create one dynamically."""

        # Try to use pre-computed topic index first
        if (
            winning_topic in self.topic_bm25_indexes
            and winning_topic in self.topic_docs_mapping
        ):
            print(f"📚 Using pre-computed BM25 index for topic: {winning_topic}")
            topic_bm25 = self.topic_bm25_indexes[winning_topic]
            on_topic_docs = self.topic_docs_mapping[winning_topic]
        else:
            # Fallback: create topic index dynamically (original behavior)
            print(f"🔧 Creating dynamic BM25 index for topic: {winning_topic}")
            on_topic_docs = [
                doc for doc in self.documents if doc.get("topic_name") == winning_topic
            ]
            on_topic_texts = [doc["text"] for doc in on_topic_docs]

            if not on_topic_docs:
                on_topic_docs = self.documents
                on_topic_texts = self.document_texts

            # Create temporary BM25 index (this is what was causing the "Building index" message)
            topic_bm25 = bm25s.BM25()
            topic_tokenized_corpus = [
                self._tokenize_and_stem(text) for text in on_topic_texts
            ]
            topic_bm25.index(topic_tokenized_corpus)

        # Rerank on-topic documents using the original query for precision
        query_tokens = self._tokenize_and_stem(question)
        final_indices, _ = topic_bm25.retrieve([query_tokens], k=k)

        return [on_topic_docs[i] for i in final_indices[0]]

    def run(
        self, question: str, reference_contexts: List[str] = None
    ) -> Dict[str, Any]:
        """Executes the full 3-stage retrieval and answer generation pipeline."""
        k = 10  # Retrieve more candidates initially

        # --- Stage 1: Parallel Augmented Retrieval ---
        keyword_queries = self._generate_keyword_queries(question)
        tokenized_keyword_queries = [
            self._tokenize_and_stem(query) for query in keyword_queries
        ]
        bm25_results_indices, _ = self.bm25_index.retrieve(
            tokenized_keyword_queries, k=k
        )

        semantic_statements = self._augment_semantic_statements(question)
        query_embeddings = self.embedding_model.encode(
            semantic_statements, convert_to_numpy=True
        )
        _, faiss_results_indices = self._safe_faiss_search(query_embeddings, k)

        # Combine and unique candidates
        candidate_indices = set()
        for indices in bm25_results_indices:
            candidate_indices.update(indices)
        for indices in faiss_results_indices:
            candidate_indices.update(indices)

        candidate_docs = [self.documents[i] for i in candidate_indices]

        # --- Stage 2: Topic Selection via Majority Vote ---
        if not candidate_docs:
            winning_topic = "Unknown"
        else:
            topic_counts = Counter(
                doc.get("topic_name", "Unknown") for doc in candidate_docs
            )
            winning_topic = topic_counts.most_common(1)[0][0]

        # --- Stage 3: Optimized Focused BM25 Reranking ---
        retrieved_docs = self._get_topic_specific_results(winning_topic, question, k=5)

        # --- Final Answer Generation ---
        context = "\n".join([doc["text"] for doc in retrieved_docs])
        llm_response = self.llm_client.classify_statement(question, context)

        answer = {
            "statement_is_true": llm_response[0],
            "statement_topic": self.llm_client._topic_name_to_number(winning_topic),
        }

        return {
            "answer": json.dumps(answer),
            "context": [doc["text"] for doc in retrieved_docs],
        }

    def _topic_name_to_number(self, topic_name: str) -> int:
        """Convert topic name to number using the LLM client's mapping."""
        return self.llm_client._topic_name_to_number(topic_name)
