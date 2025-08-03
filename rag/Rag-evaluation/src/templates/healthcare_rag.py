"""
Emergency Healthcare RAG template using BM25s for better retrieval and Qwen 8B via Ollama.
"""

import os
import json
import pickle
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Any
import bm25s
import Stemmer  # PyStemmer for stemming

try:
    from ..llm_client import LocalLLMClient
except ImportError:
    # Fallback for direct execution
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from llm_client import LocalLLMClient


class HealthcareRAG:
    """RAG system specialized for emergency healthcare questions using BM25s similarity."""

    def __init__(self, llm_client: LocalLLMClient = None):
        self.llm_client = llm_client or LocalLLMClient()
        self.documents = []
        self.document_texts = []
        self.tokenized_corpus = None
        self.bm25_index = None
        self.stemmer = Stemmer.Stemmer("english")  # English stemmer
        self._setup_documents()

    def _setup_documents(self):
        """Initialize document collection with medical documents using cache."""
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)

        # Check if cached data exists and is valid
        if self._load_from_cache(cache_dir):
            print(
                f"✅ Loaded cached BM25s index for {len(self.document_texts)} documents"
            )
            return

        print("🔄 Cache not found or invalid. Building BM25s index...")
        # Load medical documents
        self._load_medical_documents()

        if self.document_texts:
            # Tokenize and stem the corpus
            print("📝 Tokenizing and stemming documents...")
            self.tokenized_corpus = [
                self._tokenize_and_stem(doc) for doc in self.document_texts
            ]

            # Create BM25s index
            print("🔍 Building BM25s index...")
            self.bm25_index = bm25s.BM25()
            self.bm25_index.index(self.tokenized_corpus)
            print(f"Created BM25s index for {len(self.document_texts)} documents")

            # Save to cache
            self._save_to_cache(cache_dir)
            print("💾 Cached BM25s index for future use")

    def _load_medical_documents(self):
        """Load and process medical documents from topics directory and training data."""
        # Load training data first - this provides statement-answer patterns
        self._load_training_data()

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

        print(
            f"Loaded {len(self.documents)} document chunks (including training examples)"
        )

    def _load_training_data(self):
        """Load training statements and answers to improve context retrieval."""
        # Find training data directory
        training_paths = [
            "data/train",
            "../data/train",
            "../../data/train",
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "train"),
        ]

        train_dir = None
        for path in training_paths:
            if os.path.exists(path):
                train_dir = path
                break

        if not train_dir:
            print("Warning: Training data directory not found")
            return

        statements_dir = os.path.join(train_dir, "statements")
        answers_dir = os.path.join(train_dir, "answers")

        if not os.path.exists(statements_dir) or not os.path.exists(answers_dir):
            print("Warning: Training statements or answers directory not found")
            return

        # Load training examples
        training_count = 0
        topic_name_mapping = {v: k for k, v in self.llm_client.topic_mapping.items()}

        for filename in os.listdir(statements_dir):
            if filename.endswith(".txt"):
                statement_file = os.path.join(statements_dir, filename)
                answer_file = os.path.join(
                    answers_dir, filename.replace(".txt", ".json")
                )

                if os.path.exists(answer_file):
                    try:
                        # Read statement
                        with open(statement_file, "r", encoding="utf-8") as f:
                            statement = f.read().strip()

                        # Read answer
                        with open(answer_file, "r", encoding="utf-8") as f:
                            answer_data = json.load(f)

                        # Get topic name from number
                        topic_num = answer_data.get("statement_topic", 0)
                        topic_name = topic_name_mapping.get(topic_num, "Unknown")
                        is_true = answer_data.get("statement_is_true", 1)

                        # Create enhanced training document
                        training_doc = f"""MEDICAL STATEMENT: {statement}
TRUTH VALUE: {"TRUE" if is_true else "FALSE"}
TOPIC: {topic_name}
TOPIC_NUMBER: {topic_num}
TRAINING_EXAMPLE: This statement is {"true" if is_true else "false"} and relates to {topic_name}."""

                        self.documents.append(
                            {
                                "content": training_doc,
                                "source": f"training_{filename}",
                                "topic": topic_name,
                                "topic_number": topic_num,
                                "is_training": True,
                                "original_statement": statement,
                            }
                        )
                        self.document_texts.append(training_doc)
                        training_count += 1

                    except Exception as e:
                        print(f"Error loading training example {filename}: {e}")

        print(f"Loaded {training_count} training examples")

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

    def _tokenize_and_stem(self, text: str) -> List[str]:
        """Tokenize and stem text for BM25s processing."""
        # Simple tokenization - split on whitespace and punctuation
        tokens = re.findall(r"\b\w+\b", text.lower())

        # Stem tokens for better matching
        stemmed_tokens = self.stemmer.stemWords(tokens)

        return stemmed_tokens

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
        """Load documents and BM25s index from cache if valid."""
        try:
            cache_file = cache_dir / "bm25s_cache.pkl"
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
            self.tokenized_corpus = cache_data["tokenized_corpus"]
            self.bm25_index = cache_data["bm25_index"]

            return True

        except Exception as e:
            print(f"⚠️ Error loading cache: {e}")
            return False

    def _save_to_cache(self, cache_dir: Path) -> None:
        """Save documents and BM25s index to cache."""
        try:
            cache_file = cache_dir / "bm25s_cache.pkl"
            hash_file = cache_dir / "data_hash.txt"

            # Save cache data
            cache_data = {
                "documents": self.documents,
                "document_texts": self.document_texts,
                "tokenized_corpus": self.tokenized_corpus,
                "bm25_index": self.bm25_index,
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
        """Retrieve relevant medical contexts for the query using BM25s similarity."""
        if self.bm25_index is None or not self.document_texts:
            return []

        try:
            # Preprocess query for better matching
            processed_query = self._preprocess_query(query)

            # Tokenize and stem the query
            query_tokens = self._tokenize_and_stem(processed_query)

            # Get BM25 scores for fewer documents to speed up
            # BM25s expects tokenized query as list of lists
            results = self.bm25_index.retrieve([query_tokens], k=k)
            top_indices = results.documents[0]  # Get first query's document indices

            # Return the text content of top k documents
            contexts = [self.document_texts[i] for i in top_indices]

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

            # Combine contexts - but limit total length to avoid overwhelming the LLM
            # Take top 5 most relevant contexts and limit total length
            limited_contexts = [ctx[:200] for ctx in retrieved_contexts[:5]]
            combined_context = "\n".join(limited_contexts) if limited_contexts else ""

            # Use LLM to classify the statement
            statement_is_true, statement_topic = self.llm_client.classify_statement(
                question, combined_context
            )

            # Format answer - only include required fields for leaderboard
            answer = {
                "statement_is_true": statement_is_true,
                "statement_topic": statement_topic,
            }

            return {"answer": json.dumps(answer), "context": retrieved_contexts}

        except Exception as e:
            print(f"Error in HealthcareRAG.run: {e}")
            # Return safe defaults
            return {
                "answer": json.dumps({"statement_is_true": 1, "statement_topic": 0}),
                "context": [],
            }
