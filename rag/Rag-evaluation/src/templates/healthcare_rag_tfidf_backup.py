"""
Emergency Healthcare RAG template using Qwen 8B via Ollama.
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from ..llm_client import LocalLLMClient


class HealthcareRAG:
    """RAG system specialized for emergency healthcare questions using TF-IDF similarity."""

    def __init__(self, llm_client: LocalLLMClient = None):
        self.llm_client = llm_client or LocalLLMClient()
        self.documents = []
        self.document_texts = []
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=15000,  # Increased for better coverage
            ngram_range=(1, 2),  # Include bigrams for better matching
            min_df=1,  # Include rare terms
            max_df=0.95,  # Remove very common terms
        )
        self.tfidf_matrix = None
        self._setup_documents()

    def _setup_documents(self):
        """Initialize document collection with medical documents using cache."""
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)

        # Check if cached data exists and is valid
        if self._load_from_cache(cache_dir):
            print(
                f"✅ Loaded cached TF-IDF matrix for {len(self.document_texts)} documents"
            )
            return

        print("🔄 Cache not found or invalid. Building TF-IDF matrix...")
        # Load medical documents
        self._load_medical_documents()

        if self.document_texts:
            # Create TF-IDF matrix
            self.tfidf_matrix = self.vectorizer.fit_transform(self.document_texts)
            print(f"Created TF-IDF matrix for {len(self.document_texts)} documents")

            # Save to cache
            self._save_to_cache(cache_dir)
            print("💾 Cached TF-IDF matrix for future use")

    def _load_medical_documents(self):
        """Load and process medical documents from topics directory."""
        # Find topics directory
        topics_dir = self._find_topics_directory()
        if not topics_dir:
            print("Warning: Topics directory not found")
            return

        # Process all markdown files
        for root, dirs, files in os.walk(topics_dir):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        if content.strip():  # Only add non-empty documents
                            # Split content into chunks for better retrieval
                            chunks = self._split_text(
                                content, chunk_size=1000, chunk_overlap=200
                            )

                            for i, chunk in enumerate(chunks):
                                if chunk.strip():
                                    self.documents.append(
                                        {
                                            "content": chunk,
                                            "source": file_path,
                                            "topic": os.path.basename(
                                                os.path.dirname(file_path)
                                            ),
                                            "chunk_id": i,
                                        }
                                    )
                                    self.document_texts.append(chunk)

                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

        print(f"Loaded {len(self.documents)} document chunks")

    def _split_text(
        self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[str]:
        """Simple text splitter."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                # Try to split at sentence boundary
                last_period = text.rfind(".", start, end)
                if last_period > start + chunk_size // 2:
                    end = last_period + 1

            chunks.append(text[start:end])
            start = end - chunk_overlap if end < len(text) else end

        return chunks

    def _find_topics_directory(self) -> str:
        """Find the topics directory."""
        possible_paths = [
            "data/topics",
            "../data/topics",
            "../../data/topics",
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "topics"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def _get_data_hash(self) -> str:
        """Generate hash of topics directory to detect changes."""
        topics_dir = self._find_topics_directory()
        if not topics_dir:
            return ""

        hasher = hashlib.md5()
        # Include directory modification time and file count
        try:
            for root, dirs, files in os.walk(topics_dir):
                for file in sorted(files):  # Sort for consistent hashing
                    if file.endswith(".md"):
                        file_path = os.path.join(root, file)
                        stat = os.stat(file_path)
                        hasher.update(
                            f"{file_path}:{stat.st_mtime}:{stat.st_size}".encode()
                        )
        except Exception:
            return ""

        return hasher.hexdigest()

    def _load_from_cache(self, cache_dir: Path) -> bool:
        """Load documents and TF-IDF matrix from cache if valid."""
        try:
            cache_file = cache_dir / "tfidf_cache.pkl"
            hash_file = cache_dir / "data_hash.txt"

            if not cache_file.exists() or not hash_file.exists():
                return False

            # Check if data has changed
            current_hash = self._get_data_hash()
            with open(hash_file, "r") as f:
                cached_hash = f.read().strip()

            if current_hash != cached_hash:
                print("📁 Data files have changed, rebuilding cache...")
                return False

            # Load cached data
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            self.documents = cache_data["documents"]
            self.document_texts = cache_data["document_texts"]
            self.vectorizer = cache_data["vectorizer"]
            self.tfidf_matrix = cache_data["tfidf_matrix"]

            return True

        except Exception as e:
            print(f"⚠️ Error loading cache: {e}")
            return False

    def _save_to_cache(self, cache_dir: Path) -> None:
        """Save documents and TF-IDF matrix to cache."""
        try:
            cache_file = cache_dir / "tfidf_cache.pkl"
            hash_file = cache_dir / "data_hash.txt"

            # Save cache data
            cache_data = {
                "documents": self.documents,
                "document_texts": self.document_texts,
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
            }

            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)

            # Save data hash
            current_hash = self._get_data_hash()
            with open(hash_file, "w") as f:
                f.write(current_hash)

        except Exception as e:
            print(f"⚠️ Error saving cache: {e}")

    def _preprocess_query(self, query: str) -> str:
        """Preprocess query to improve retrieval."""
        # Expand medical abbreviations and add synonyms
        expansions = {
            "CHD": "coronary heart disease",
            "MI": "myocardial infarction heart attack",
            "CPR": "cardiopulmonary resuscitation",
            "ACS": "acute coronary syndrome",
            "PE": "pulmonary embolism",
            "HTN": "hypertension high blood pressure",
            "DM": "diabetes mellitus",
            "COPD": "chronic obstructive pulmonary disease",
            "DVT": "deep vein thrombosis",
            "ICU": "intensive care unit",
            "ED": "emergency department",
            "EKG": "electrocardiogram ECG",
            "BP": "blood pressure",
            "HR": "heart rate",
        }

        expanded_query = query
        for abbrev, expansion in expansions.items():
            if abbrev in query.upper():
                expanded_query += f" {expansion}"

        return expanded_query

    def retrieve_context(self, query: str, k: int = 15) -> List[str]:
        """Retrieve relevant medical contexts for the query using improved TF-IDF similarity."""
        if self.tfidf_matrix is None or not self.document_texts:
            return []

        try:
            # Preprocess query for better matching
            processed_query = self._preprocess_query(query)

            # Transform query to TF-IDF vector
            query_vector = self.vectorizer.transform([processed_query])

            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

            # Get top k most similar documents (NO minimum threshold)
            top_indices = np.argsort(similarities)[-k:][::-1]

            # Return the text content of top k documents regardless of similarity
            contexts = [self.document_texts[i] for i in top_indices[:k]]

            return contexts
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []

    def run(
        self, question: str, reference_contexts: List[str] = None
    ) -> Dict[str, Any]:
        """
        Process a medical statement question.

        Args:
            question: The medical statement to evaluate
            reference_contexts: Optional reference contexts (not used in this implementation)

        Returns:
            Dict with 'answer' and 'context' keys
        """
        try:
            # Retrieve relevant contexts
            retrieved_contexts = self.retrieve_context(question)

            # Combine contexts - limit to first 300 chars of each for speed
            limited_contexts = [ctx[:300] for ctx in retrieved_contexts]
            combined_context = "\n".join(limited_contexts) if limited_contexts else ""

            # Use LLM to classify the statement
            statement_is_true, statement_topic = self.llm_client.classify_statement(
                question, combined_context
            )

            # Format answer
            answer = {
                "statement_is_true": statement_is_true,
                "statement_topic": statement_topic,
                "explanation": f"Based on medical knowledge and context, this statement is {'true' if statement_is_true else 'false'} and relates to topic {statement_topic}.",
            }

            return {"answer": json.dumps(answer), "context": retrieved_contexts}

        except Exception as e:
            print(f"Error in HealthcareRAG.run: {e}")
            # Return safe defaults
            return {
                "answer": json.dumps(
                    {
                        "statement_is_true": 1,
                        "statement_topic": 0,
                        "explanation": "Error occurred during processing",
                    }
                ),
                "context": [],
            }
