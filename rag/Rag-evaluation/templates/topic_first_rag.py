"""
Topic-First Sequential RAG Template

Smart two-stage pipeline:
1. Topic classification with diverse context (3-4s)
2. Binary classification with topic-filtered context (2-3s)

Expected: 90%+ binary, 80%+ topic, 6-7s total time
"""

import json
import logging
import time
from typing import Dict, List, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from llm_client import LocalLLMClient
from contextlib import contextmanager


@contextmanager
def timed_step(step_name):
    import time

    start = time.perf_counter()

    class Timer:
        def log(self, message):
            print(f"  {message}")

    timer = Timer()
    yield timer
    end = time.perf_counter()
    print(f"{step_name} took {end - start:.4f} seconds")


logger = logging.getLogger(__name__)


class TopicFirstRAG:
    """Topic-first sequential RAG with smart two-stage classification."""

    def __init__(self, llm_client=None, **kwargs):
        self.llm_client = llm_client or LocalLLMClient()

        # Initialize components
        self.biobert_model = None
        self.faiss_index = None
        self.bm25_index = None
        self.document_mapping = None

        # Load official topic mapping
        self.topic_mapping = self._load_topic_mapping()

        # Initialize indexes path
        self.optimized_indexes_path = "optimized_indexes"

        logger.info("TopicFirstRAG initialized")

    def _load_topic_mapping(self) -> Dict[str, int]:
        """Load official topic mapping from topics.json."""
        try:
            import os

            current_dir = os.path.dirname(__file__)
            topics_path = os.path.join(current_dir, "..", "..", "data", "topics.json")
            with open(topics_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not load topic mapping: {e}")
            return {}

    def initialize(self, optimized_indexes_path: str) -> None:
        """Initialize with pre-computed indexes."""
        with timed_step("Loading essential indexes") as timer:
            # Load BioBERT FAISS index
            faiss_path = f"{optimized_indexes_path}/biobert_faiss.index"
            self.faiss_index = faiss.read_index(faiss_path)
            timer.log(
                f"✅ Loaded BioBERT FAISS index ({self._get_file_size(faiss_path)})"
            )

            # Load BM25 index
            bm25_path = f"{optimized_indexes_path}/bm25_index.pkl"
            import pickle

            with open(bm25_path, "rb") as f:
                self.bm25_index = pickle.load(f)
            timer.log(f"✅ Loaded global BM25 index ({self._get_file_size(bm25_path)})")

            # Load document mapping
            mapping_path = f"{optimized_indexes_path}/document_mapping.json"
            with open(mapping_path, "r") as f:
                self.document_mapping = json.load(f)
            doc_count = len(self.document_mapping)
            timer.log(
                f"✅ Loaded document mapping ({doc_count} docs, {self._get_file_size(mapping_path)})"
            )

        with timed_step("Loading BioBERT model") as timer:
            self.biobert_model = SentenceTransformer(
                "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
                device="mps" if hasattr(faiss, "StandardGpuResources") else "cpu",
            )
            timer.log("🧠 BioBERT model loaded")

        logger.info("✅ TopicFirstRAG ready with sequential topic-first classification")

    def _get_file_size(self, file_path: str) -> str:
        """Get human-readable file size."""
        import os

        size_bytes = os.path.getsize(file_path)
        if size_bytes < 1024**2:
            return f"{size_bytes / 1024:.1f}KB"
        else:
            return f"{size_bytes / (1024**2):.1f}MB"

    def _fast_hybrid_retrieval(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        """Fast hybrid retrieval combining BioBERT and BM25."""
        with timed_step("BioBERT Encoding"):
            query_embedding = self.biobert_model.encode([query])

        with timed_step("FAISS Search"):
            faiss_scores, faiss_indices = self.faiss_index.search(
                query_embedding.astype("float32"), k * 2
            )

        with timed_step("BM25 Search"):
            bm25_scores = self.bm25_index.get_scores(query.lower().split())
            bm25_indices = np.argsort(bm25_scores)[::-1][: k * 2]

        with timed_step("Document Assembly"):
            # Combine and deduplicate
            combined_indices = []
            seen_doc_ids = set()

            # Add FAISS results with scores
            for i, idx in enumerate(faiss_indices[0]):
                if idx < len(self.document_mapping):
                    doc_id = str(idx)
                    if doc_id not in seen_doc_ids:
                        doc = self.document_mapping[idx].copy()
                        doc["faiss_score"] = float(faiss_scores[0][i])
                        doc["bm25_score"] = (
                            float(bm25_scores[idx]) if idx < len(bm25_scores) else 0.0
                        )
                        doc["doc_id"] = doc_id
                        combined_indices.append(doc)
                        seen_doc_ids.add(doc_id)

            # Add top BM25 results if not already included
            for idx in bm25_indices:
                if idx < len(self.document_mapping):
                    doc_id = str(idx)
                    if doc_id not in seen_doc_ids and len(combined_indices) < k:
                        doc = self.document_mapping[idx].copy()
                        doc["faiss_score"] = 0.0
                        doc["bm25_score"] = float(bm25_scores[idx])
                        doc["doc_id"] = doc_id
                        combined_indices.append(doc)
                        seen_doc_ids.add(doc_id)

            return combined_indices[:k]

    def _build_diverse_context(self, candidate_docs: List[Dict[str, Any]]) -> str:
        """Build diverse context for topic classification from multiple domains."""
        selected_chunks = []

        for doc in candidate_docs:
            # Select best chunks from each document for diversity
            chunks = doc.get("chunks", [])[:3]  # Top 3 chunks per doc
            for chunk in chunks:
                selected_chunks.append(
                    {
                        "text": chunk.get("text", ""),
                        "topic_name": doc.get("topic_name"),
                        "doc_id": doc.get("doc_id"),
                    }
                )

        # Sort by relevance and limit total
        context_parts = []
        for i, chunk in enumerate(
            selected_chunks[:12]
        ):  # Diverse context up to 12 chunks
            topic = chunk["topic_name"]
            text = chunk["text"][:800]  # Limit chunk size
            context_parts.append(f"[{topic}]: {text}")

        return "\n\n".join(context_parts)

    def _build_focused_context(self, topic_filtered_docs: List[Dict[str, Any]]) -> str:
        """Build focused context for binary classification from topic-specific docs."""
        selected_chunks = []

        for doc in topic_filtered_docs:
            chunks = doc.get("chunks", [])
            # Score chunks for binary classification
            scored_chunks = []
            for chunk in chunks:
                score = self._score_chunk_for_binary(chunk.get("text", ""))
                scored_chunks.append((score, chunk))

            # Sort and take top chunks
            scored_chunks.sort(reverse=True, key=lambda x: x[0])
            selected_chunks.extend(
                [chunk for _, chunk in scored_chunks[:4]]
            )  # Top 4 per doc

        # Build focused context
        context_parts = []
        for chunk in selected_chunks[:8]:  # Focused context up to 8 chunks
            text = chunk.get("text", "")[:600]  # Smaller chunks for focused context
            context_parts.append(text)

        return "\n\n".join(context_parts)

    def _score_chunk_for_binary(self, text: str) -> float:
        """Score chunks for binary classification relevance."""
        text_lower = text.lower()

        # Factual indicator bonus
        factual_indicators = [
            "study",
            "research",
            "clinical",
            "evidence",
            "trial",
            "data",
            "results",
        ]
        factual_score = (
            sum(1 for indicator in factual_indicators if indicator in text_lower) * 0.5
        )

        # Length penalty (prefer substantial but not overly long chunks)
        length_penalty = max(0, 1.0 - abs(len(text) - 400) / 400) * 0.2

        return factual_score + length_penalty

    def classify_topic_only(self, statement: str, diverse_context: str) -> int:
        """Stage 1: Topic classification with diverse context."""
        try:
            _, topic_number = self.llm_client.classify_statement(
                statement, diverse_context
            )
            return topic_number

        except Exception as e:
            logger.error(f"Topic classification error: {e}")
            return 0  # Default fallback

    def classify_binary_only(self, statement: str, focused_context: str) -> bool:
        """Stage 2: Binary classification with topic-focused context."""
        try:
            is_true, _ = self.llm_client.classify_statement(statement, focused_context)
            return bool(is_true)

        except Exception as e:
            logger.error(f"Binary classification error: {e}")
            return False  # Default fallback

    def topic_first_classification(self, question: str) -> Tuple[bool, int]:
        """Smart two-stage topic-first classification."""
        start_time = time.time()

        # Stage 1: Fast hybrid retrieval (0.03s)
        with timed_step("Hybrid Retrieval"):
            candidate_docs = self._fast_hybrid_retrieval(question, k=8)

        # Stage 2: Topic classification with diverse context (3-4s)
        with timed_step("Context Preparation"):
            topic_context = self._build_diverse_context(candidate_docs)
            print(f"Topic context: {len(topic_context)} chars")

        with timed_step("Topic Classification"):
            topic_number = self.classify_topic_only(question, topic_context)

        # Stage 3: Filter documents to the classified topic
        with timed_step("Topic Filtering"):
            topic_filtered_docs = [
                doc
                for doc in candidate_docs
                if self.topic_mapping.get(doc.get("topic_name")) == topic_number
            ]

            # If no docs in that topic, fall back to most relevant
            if not topic_filtered_docs:
                topic_filtered_docs = candidate_docs[:3]
                print(f"No docs found for topic {topic_number}, using top 3 candidates")
            else:
                print(f"Found {len(topic_filtered_docs)} docs for topic {topic_number}")

        # Stage 4: Binary classification with topic-focused context (2-3s)
        with timed_step("Focused Context Preparation"):
            binary_context = self._build_focused_context(topic_filtered_docs)
            print(f"Binary context: {len(binary_context)} chars")

        with timed_step("Binary Classification"):
            is_true = self.classify_binary_only(question, binary_context)

        with timed_step("Response Formatting"):
            total_time = time.time() - start_time
            print(f"Topic-first classification completed in {total_time:.2f}s")

        return is_true, topic_number

    def run(
        self, question: str, reference_contexts: list[str] = None
    ) -> dict[str, any]:
        """Main run method for evaluation framework."""
        # Initialize if not already done
        if self.biobert_model is None:
            self.initialize(self.optimized_indexes_path)

        is_true, topic_number = self.topic_first_classification(question)
        return {
            "answer": json.dumps(
                {"statement_is_true": int(is_true), "statement_topic": topic_number}
            ),
            "context": [],
        }
