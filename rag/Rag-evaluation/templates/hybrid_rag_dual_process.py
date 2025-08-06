"""
Hybrid RAG template with dual parallel processes - optimized version
Uses separate processes for binary and topic classification with specialized contexts
"""

import json
import pickle
import os
import re
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor

# Fix OpenMP library conflicts on ARM Mac
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import faiss
import Stemmer
from sentence_transformers import SentenceTransformer

# Configure FAISS for ARM Mac stability
try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass

from llm_client import LocalLLMClient


@contextmanager
def timed_step(step_name):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{step_name} took {end - start:.4f} seconds")


def binary_classification_worker(
    question: str, context: str, model_name: str = "cogito:3b"
):
    """Worker process for binary classification - focused on TRUE/FALSE only."""
    try:
        import requests

        # Simplified binary prompt
        prompt = f"""[SYSTEM]
You are a medical fact-checker. Determine if the statement is TRUE or FALSE based ONLY on the provided context.

[CONTEXT]
{context}

[STATEMENT]
"{question}"

[INSTRUCTIONS]
- If context SUPPORTS the statement → TRUE (1)
- If context CONTRADICTS the statement → FALSE (0)
- If context lacks info → Make best judgment

[OUTPUT]
JSON only: {{"is_true": 1}} or {{"is_true": 0}}
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 0.1,
                    "top_k": 1,
                    "num_predict": 30,
                    "repeat_penalty": 1.0,
                },
            },
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json().get("response", "")
            try:
                parsed = json.loads(result)
                return max(0, min(1, int(parsed.get("is_true", 1))))
            except Exception:
                return 1
        return 1

    except Exception as e:
        print(f"Binary classification error: {e}")
        return 1


def topic_classification_worker(
    question: str, context: str, topic_mapping: Dict, model_name: str = "cogito:3b"
):
    """Worker process for topic classification - focused on categorization."""
    try:
        import requests

        # Format topics for prompt
        topics_text = "\n".join(
            f"- {name}: {num}" for name, num in topic_mapping.items()
        )

        prompt = f"""[SYSTEM]
You are a medical topic classifier. Classify the statement into the most appropriate medical topic.

[CONTEXT]
{context}

[STATEMENT]
"{question}"

[TOPICS]
{topics_text}

[INSTRUCTIONS]
Choose the topic number that best matches the statement's medical domain.

[OUTPUT]
JSON only: {{"topic": NUMBER}}
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p": 0.1,
                    "top_k": 1,
                    "num_predict": 30,
                    "repeat_penalty": 1.0,
                },
            },
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json().get("response", "")
            try:
                parsed = json.loads(result)
                topic = int(parsed.get("topic", 0))
                return max(0, min(114, topic))
            except Exception:
                return 0
        return 0

    except Exception as e:
        print(f"Topic classification error: {e}")
        return 0


class HybridRAGDualProcess:
    """Hybrid RAG with dual parallel processes for binary and topic classification."""

    def __init__(self, llm_client: LocalLLMClient = None):
        self.llm_client = llm_client or LocalLLMClient()
        self.stemmer = Stemmer.Stemmer("english")
        self.model_name = self.llm_client.model_name

        # Core optimized indexes
        self.documents = []
        self.document_texts = []
        self.biobert_faiss_index = None
        self.global_bm25_index = None
        self.biobert_model = None

        # Load optimized indexes
        self._load_optimized_indexes()

        print("✅ HybridRAGDualProcess ready with parallel classification")

    def _load_optimized_indexes(self):
        """Load optimized indexes quickly."""
        optimized_dir = Path(__file__).parent.parent / "optimized_indexes"

        if not optimized_dir.exists():
            raise FileNotFoundError(f"Optimized indexes not found: {optimized_dir}")

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

        # Load BioBERT model
        if self.biobert_faiss_index is not None:
            print("🧠 Loading BioBERT model...")
            self.biobert_model = SentenceTransformer(
                "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
                device="mps" if hasattr(faiss, "StandardGpuResources") else "cpu",
            )

    def _tokenize_and_stem(self, text: str) -> List[str]:
        """Tokenize and stem text."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        return self.stemmer.stemWords(tokens)

    def _safe_faiss_search(
        self, query_embeddings: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Safely perform FAISS search."""
        try:
            if not isinstance(query_embeddings, np.ndarray):
                query_embeddings = np.array(query_embeddings)

            query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)

            if query_embeddings.ndim == 1:
                query_embeddings = query_embeddings.reshape(1, -1)

            expected_dim = self.biobert_faiss_index.d
            if query_embeddings.shape[1] != expected_dim:
                raise ValueError(
                    f"Query embedding dimension mismatch: {query_embeddings.shape[1]} vs {expected_dim}"
                )

            distances, indices = self.biobert_faiss_index.search(query_embeddings, k)
            return distances, indices

        except Exception as e:
            print(f"⚠️  FAISS search failed: {e}")
            return np.array([[]]), np.array([[]])

    def _select_best_chunks(
        self, question: str, candidate_docs: List[Dict], max_chunks: int = 3
    ) -> List[Dict]:
        """Select most relevant chunks for binary classification."""
        if len(candidate_docs) <= max_chunks:
            return candidate_docs

        question_words = set(self._tokenize_and_stem(question))
        question_raw_words = set(question.lower().split())

        scored_chunks = []
        for doc in candidate_docs:
            # Tokenized overlap
            doc_words = set(self._tokenize_and_stem(doc["text"]))
            stemmed_overlap = len(question_words.intersection(doc_words))

            # Raw word overlap
            doc_raw_words = set(doc["text"].lower().split())
            raw_overlap = len(question_raw_words.intersection(doc_raw_words))

            # Medical keywords
            medical_terms = [
                "acute",
                "chronic",
                "syndrome",
                "disease",
                "diagnosis",
                "treatment",
                "patient",
                "clinical",
                "symptoms",
                "therapy",
                "medical",
                "condition",
                "pain",
                "fever",
                "infection",
                "cardiac",
                "pulmonary",
                "renal",
            ]
            medical_overlap = sum(
                1
                for term in medical_terms
                if term in question.lower() and term in doc["text"].lower()
            )

            # Length penalty
            length_penalty = min(len(doc["text"]) // 500, 3)

            # Combined score
            total_score = (
                stemmed_overlap * 2  # Stemmed words
                + raw_overlap  # Raw words
                + medical_overlap * 3  # Medical terms
                - length_penalty  # Length penalty
            )

            scored_chunks.append((total_score, doc))

        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scored_chunks[:max_chunks]]

    def _select_diverse_chunks(
        self, candidate_docs: List[Dict], max_chunks: int = 4
    ) -> List[Dict]:
        """Select diverse chunks for topic classification."""
        if len(candidate_docs) <= max_chunks:
            return candidate_docs

        # Try to get diverse chunks by article titles and content
        selected = []
        seen_titles = set()

        # First pass: unique article titles
        for doc in candidate_docs:
            title = doc.get("article_title", "Unknown")
            if title not in seen_titles:
                selected.append(doc)
                seen_titles.add(title)
                if len(selected) >= max_chunks:
                    break

        # If we don't have enough, fill with remaining docs
        if len(selected) < max_chunks:
            remaining = [doc for doc in candidate_docs if doc not in selected]
            selected.extend(remaining[: max_chunks - len(selected)])

        return selected[:max_chunks]

    def _fast_hybrid_retrieval(self, question: str, k: int = 10) -> List[Dict]:
        """Fast hybrid retrieval using BioBERT FAISS + Global BM25."""
        candidate_indices = set()

        # BioBERT semantic search
        if self.biobert_faiss_index is not None and self.biobert_model is not None:
            with timed_step("  BioBERT Encoding"):
                try:
                    query_embedding = self.biobert_model.encode([question])
                except Exception as e:
                    print(f"⚠️  BioBERT encoding failed: {e}")
                    query_embedding = None

            if query_embedding is not None:
                with timed_step("  FAISS Search"):
                    try:
                        _, faiss_indices = self._safe_faiss_search(query_embedding, k)
                        candidate_indices.update(faiss_indices[0])
                    except Exception as e:
                        print(f"⚠️  FAISS search failed: {e}")

        # Global BM25 keyword search
        if self.global_bm25_index is not None:
            with timed_step("  BM25 Search"):
                try:
                    query_tokens = self._tokenize_and_stem(question)
                    bm25_results = self.global_bm25_index.retrieve([query_tokens], k=k)
                    candidate_indices.update(bm25_results.documents[0])
                except Exception as e:
                    print(f"⚠️  BM25 search failed: {e}")

        # Return candidate documents
        with timed_step("  Document Assembly"):
            candidate_docs = [
                self.documents[i] for i in candidate_indices if i < len(self.documents)
            ]

        return candidate_docs[:k]

    def run(
        self, question: str, reference_contexts: List[str] = None
    ) -> Dict[str, Any]:
        """
        Process using dual parallel processes for binary and topic classification.
        """
        try:
            # Fast hybrid retrieval
            with timed_step("Hybrid Retrieval"):
                candidate_docs = self._fast_hybrid_retrieval(question, k=10)

            # Prepare specialized contexts
            with timed_step("Context Preparation"):
                # Binary context: focused, relevant chunks
                binary_chunks = self._select_best_chunks(
                    question, candidate_docs, max_chunks=2
                )
                binary_context = "\n\n".join(
                    [
                        f"[{doc.get('article_title', 'Medical Text')}]: {doc['text']}"
                        for doc in binary_chunks
                    ]
                )

                # Topic context: diverse, comprehensive chunks
                topic_chunks = self._select_diverse_chunks(candidate_docs, max_chunks=4)
                topic_context = "\n\n".join(
                    [
                        f"[{doc.get('article_title', 'Medical Text')}]: {doc['text']}"
                        for doc in topic_chunks
                    ]
                )

            print(
                f"Binary context: {len(binary_context)} chars, Topic context: {len(topic_context)} chars"
            )

            # Run parallel processes
            with timed_step("Parallel Classification"):
                with ProcessPoolExecutor(max_workers=2) as executor:
                    # Submit both tasks
                    binary_future = executor.submit(
                        binary_classification_worker,
                        question,
                        binary_context,
                        self.model_name,
                    )
                    topic_future = executor.submit(
                        topic_classification_worker,
                        question,
                        topic_context,
                        self.llm_client.topic_mapping,
                        self.model_name,
                    )

                    # Wait for both results
                    statement_is_true = binary_future.result(timeout=60)
                    statement_topic = topic_future.result(timeout=60)

            with timed_step("Response Formatting"):
                answer = {
                    "statement_is_true": statement_is_true,
                    "statement_topic": statement_topic,
                }

                # Return contexts from both processes
                retrieved_contexts = [
                    doc["text"] for doc in binary_chunks + topic_chunks
                ]

            return {"answer": json.dumps(answer), "context": retrieved_contexts}

        except Exception as e:
            print(f"Error in HybridRAGDualProcess.run: {e}")
            import traceback

            traceback.print_exc()

            # Return safe defaults
            return {
                "answer": json.dumps({"statement_is_true": 1, "statement_topic": 0}),
                "context": [],
            }
