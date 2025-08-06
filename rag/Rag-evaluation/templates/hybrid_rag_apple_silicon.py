"""
Hybrid RAG template for Apple Silicon - fast version using optimized data sources
Uses global BM25 and BioBERT FAISS for speed while leveraging pre-built indexes
"""

import json
import pickle
import os
import re
import time
import numpy as np
import requests
from pathlib import Path
from typing import Dict, List, Any, Tuple
from contextlib import contextmanager

# Fix OpenMP library conflicts on ARM Mac
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import faiss
import Stemmer
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

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
        self.cross_encoder = None

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

        # Load cross-encoder for reranking
        print("🔍 Loading cross-encoder for reranking...")
        try:
            self.cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                device="mps" if hasattr(faiss, "StandardGpuResources") else "cpu",
            )
            print("   ✅ Cross-encoder loaded successfully")
        except Exception as e:
            print(f"   ⚠️  Cross-encoder loading failed: {e}")
            self.cross_encoder = None

    def _tokenize_and_stem(self, text: str) -> List[str]:
        """Tokenize and stem text for BM25 processing."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        return self.stemmer.stemWords(tokens)

    def _llm_extract_keywords(self, statement: str) -> List[str]:
        """Use LLM to extract key medical concepts for BM25 search."""
        try:
            prompt = f"""Extract key medical concepts from this statement for document search.

Statement: "{statement}"

Respond with a JSON object containing a "keywords" array of 3-5 important medical terms, conditions, or concepts.

Example: {{"keywords": ["acute cholangitis", "Charcot triad", "biliary obstruction"]}}

Your response:"""

            response = requests.post(
                f"{self.llm_client.ollama_url}/api/generate",
                json={
                    "model": self.llm_client.model_name,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.1,
                        "num_predict": 100,
                    },
                },
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json().get("response", "")
                parsed = json.loads(result)
                keywords = parsed.get("keywords", [])
                return keywords[:5]  # Limit to 5 keywords

        except Exception as e:
            print(f"⚠️ LLM keyword extraction failed: {e}")

        # Fallback: simple extraction
        words = statement.split()
        return [word for word in words if len(word) > 4][:5]

    def _llm_generate_variations(self, statement: str) -> List[str]:
        """Use LLM to generate semantically similar statement variations."""
        try:
            prompt = f"""Generate alternative phrasings of this medical statement for semantic search.

Original: "{statement}"

Respond with a JSON object containing a "variations" array of 2-3 alternative statements using medical synonyms, different structures, or simplified language.

Example: {{"variations": ["Alternative phrasing 1", "Alternative phrasing 2"]}}

Your response:"""

            response = requests.post(
                f"{self.llm_client.ollama_url}/api/generate",
                json={
                    "model": self.llm_client.model_name,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.5,
                        "num_predict": 300,  # Increased token limit for complete responses
                    },
                },
                timeout=15,
            )

            if response.status_code == 200:
                result = response.json().get("response", "")

                # Clean and sanitize JSON response
                result = result.strip()
                if not result.startswith("{"):
                    # Find the first { if response has extra text
                    start = result.find("{")
                    if start != -1:
                        result = result[start:]

                # Try to fix truncated JSON by closing incomplete structures
                if not result.endswith("}"):
                    # Count open brackets and braces to try to complete the JSON
                    open_braces = result.count("{") - result.count("}")
                    open_brackets = result.count("[") - result.count("]")

                    # Try to close incomplete structures
                    if '"variations": [' in result and not result.rstrip().endswith(
                        "]"
                    ):
                        # Incomplete variations array - try to close it
                        if result.rstrip().endswith('"'):
                            result += "]}"  # Close array and object
                        elif result.rstrip().endswith(","):
                            result = (
                                result.rstrip(",") + "]}"
                            )  # Remove trailing comma and close
                        else:
                            result += '"]}'  # Close string, array, and object
                    else:
                        # Generic closure based on bracket counting
                        result += "]" * open_brackets + "}" * open_braces

                # Try to parse JSON with fallback
                try:
                    parsed = json.loads(result)
                    variations = [statement]  # Always include original
                    llm_variations = parsed.get("variations", [])

                    for variation in llm_variations:
                        if isinstance(variation, str) and len(variation) > 10:
                            # Clean the variation text
                            clean_variation = (
                                variation.strip().replace("\n", " ").replace("\r", "")
                            )
                            variations.append(clean_variation)

                    return variations[:4]  # Original + 3 variations max

                except json.JSONDecodeError as json_e:
                    print(f"    ⚠️  JSON parsing failed: {json_e}")
                    print(
                        f"    Raw response: {result[:300]}..."
                    )  # Show first 300 chars for better debugging
                    return [statement]  # Fallback to original only

        except Exception as e:
            print(f"⚠️ LLM variation generation failed: {e}")

        # Fallback: just return original
        return [statement]

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

    def _rerank_with_cross_encoder(
        self, question: str, candidate_docs: List[Dict], max_chunks: int = 3
    ) -> List[Dict]:
        """Use cross-encoder to rerank candidate documents by relevance."""
        if not self.cross_encoder or len(candidate_docs) <= max_chunks:
            return self._select_best_chunks(question, candidate_docs, max_chunks)

        try:
            # Prepare query-document pairs for cross-encoder
            pairs = [(question, doc["text"]) for doc in candidate_docs]

            # Get relevance scores
            scores = self.cross_encoder.predict(pairs)

            # Sort by relevance score
            scored_docs = list(zip(scores, candidate_docs))
            scored_docs.sort(reverse=True, key=lambda x: x[0])

            # Return top documents
            return [doc for score, doc in scored_docs[:max_chunks]]

        except Exception as e:
            print(f"⚠️ Cross-encoder reranking failed: {e}")
            # Fallback to original selection method
            return self._select_best_chunks(question, candidate_docs, max_chunks)

    def _select_best_chunks(
        self, question: str, candidate_docs: List[Dict], max_chunks: int = 3
    ) -> List[Dict]:
        """Select most relevant chunks using enhanced scoring (fallback method)."""

        if len(candidate_docs) <= max_chunks:
            return candidate_docs

        # Tokenize question for better matching
        question_words = set(self._tokenize_and_stem(question))
        question_raw_words = set(question.lower().split())

        scored_chunks = []
        for doc in candidate_docs:
            # Tokenized overlap (stemmed)
            doc_words = set(self._tokenize_and_stem(doc["text"]))
            stemmed_overlap = len(question_words.intersection(doc_words))

            # Raw word overlap
            doc_raw_words = set(doc["text"].lower().split())
            raw_overlap = len(question_raw_words.intersection(doc_raw_words))

            # Medical keywords (expanded list)
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

            # Length penalty for very long chunks (prefer concise, relevant info)
            length_penalty = min(
                len(doc["text"]) // 500, 3
            )  # Penalty for chunks >500 chars

            # Combined score
            total_score = (
                stemmed_overlap * 2  # Stemmed words are more reliable
                + raw_overlap  # Raw words catch exact matches
                + medical_overlap * 3  # Medical terms are very important
                - length_penalty  # Prefer shorter, focused chunks
            )

            scored_chunks.append((total_score, doc))

        # Return top chunks
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scored_chunks[:max_chunks]]

    def _enhanced_hybrid_retrieval(self, question: str, k: int = 10) -> List[Dict]:
        """Enhanced hybrid retrieval with multi-query approach for better coverage."""
        candidate_indices = set()

        # BM25: Multiple keyword-focused queries
        if self.global_bm25_index is not None:
            with timed_step("  BM25 Multi-Query Search"):
                try:
                    # Get medical concepts for targeted BM25 searches
                    bm25_queries = self._llm_extract_keywords(question)
                    print(f"    BM25 queries: {bm25_queries}")

                    for query in bm25_queries:
                        if query.strip():  # Only process non-empty queries
                            query_tokens = self._tokenize_and_stem(query)
                            if query_tokens:  # Only search if tokens exist
                                bm25_results = self.global_bm25_index.retrieve(
                                    [query_tokens], k=3
                                )
                                if (
                                    hasattr(bm25_results, "documents")
                                    and bm25_results.documents is not None
                                ):
                                    try:
                                        # Convert to Python list if numpy array
                                        if hasattr(bm25_results.documents, "tolist"):
                                            doc_list = bm25_results.documents.tolist()
                                        else:
                                            doc_list = bm25_results.documents

                                        # Handle nested list structure
                                        if (
                                            isinstance(doc_list, list)
                                            and len(doc_list) > 0
                                        ):
                                            if isinstance(doc_list[0], list):
                                                candidate_indices.update(doc_list[0])
                                            else:
                                                candidate_indices.update(doc_list)
                                    except Exception as inner_e:
                                        print(
                                            f"    ⚠️  BM25 result processing failed for query '{query}': {inner_e}"
                                        )
                except Exception as e:
                    print(f"⚠️  BM25 multi-query search failed: {e}")

        # Semantic: Multiple statement variations
        if self.biobert_faiss_index is not None and self.biobert_model is not None:
            with timed_step("  Semantic Multi-Query Search"):
                try:
                    # Get statement variations for semantic search
                    semantic_queries = self._llm_generate_variations(question)
                    print(f"    Semantic queries: {len(semantic_queries)} variations")

                    for query in semantic_queries:
                        if (
                            query.strip() and len(query) > 5
                        ):  # Only process meaningful queries
                            query_embedding = self.biobert_model.encode(
                                [query], normalize_embedding=True
                            )
                            _, faiss_indices = self._safe_faiss_search(
                                query_embedding, k=3
                            )
                            if faiss_indices.size > 0 and len(faiss_indices[0]) > 0:
                                candidate_indices.update(faiss_indices[0])
                except Exception as e:
                    print(f"⚠️  Semantic multi-query search failed: {e}")

        # Return candidate documents
        with timed_step("  Document Assembly"):
            candidate_docs = [
                self.documents[i] for i in candidate_indices if i < len(self.documents)
            ]
            print(f"    Found {len(candidate_docs)} candidate documents")

        return candidate_docs[:k]

    def _fast_hybrid_retrieval(self, question: str, k: int = 10) -> List[Dict]:
        """Fast hybrid retrieval using BioBERT FAISS + Global BM25 with simple fusion."""
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

        # Return candidate documents (no complex fusion)
        with timed_step("  Document Assembly"):
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
            # Enhanced hybrid retrieval using multi-query approach
            with timed_step("Enhanced Hybrid Retrieval"):
                candidate_docs = self._enhanced_hybrid_retrieval(
                    question, k=12
                )  # Get more candidates with multi-query

            # Rerank candidates with cross-encoder (or fallback to selection)
            with timed_step("Cross-Encoder Reranking"):
                selected_docs = self._rerank_with_cross_encoder(
                    question, candidate_docs, max_chunks=3
                )

            # Prepare context and get LLM classification
            with timed_step("Context Preparation"):
                context = "\n\n".join(
                    [
                        f"[{doc.get('article_title', 'Medical Text')}]: {doc['text']}"
                        for doc in selected_docs
                    ]
                )

            # Use LLM client for classification (no topic override)
            with timed_step("LLM Classification"):
                statement_is_true, statement_topic = self.llm_client.classify_statement(
                    question, context
                )

            with timed_step("Response Formatting"):
                answer = {
                    "statement_is_true": statement_is_true,
                    "statement_topic": statement_topic,
                }

                retrieved_contexts = [doc["text"] for doc in selected_docs]

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
