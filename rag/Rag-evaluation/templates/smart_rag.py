
import os
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any

import bm25s
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import Stemmer

# Fallback for running script directly
try:
    from ..llm_client import LocalLLMClient
except (ImportError, ValueError):
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from llm_client import LocalLLMClient


class SmartRAG:
    """
    A RAG template implementing a multi-stage retrieval process:
    1. Parallel Augmented Retrieval (BM25 + Semantic)
    2. Topic Selection via Majority Vote
    3. Focused BM25 Reranking within the selected topic
    """

    def __init__(self, llm_client: LocalLLMClient = None):
        self.llm_client = llm_client or LocalLLMClient()
        self.stemmer = Stemmer.Stemmer("english")
        self.documents = []
        self.document_texts = []
        self.bm25_index = None
        self.faiss_index = None
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
        self.document_texts = [doc['text'] for doc in self.documents]
        print(f"✅ Loaded {len(self.documents)} documents and indexes.")

    def _tokenize_and_stem(self, text: str) -> List[str]:
        """Basic tokenizer and stemmer for BM25."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        return self.stemmer.stemWords(tokens)

    def _generate_keyword_queries(self, query: str) -> List[str]:
        """Generates fact-focused keyword queries."""
        # Simple for now, can be expanded with entity extraction
        variations = [query]
        # Add a version with just the core nouns/adjectives
        # (A more advanced version would use PoS tagging)
        important_words = [w for w in query.split() if len(w) > 4 and w.lower() not in ["patient", "study"]]
        if important_words:
            variations.append(" ".join(important_words))
        return variations

    def _augment_semantic_statements(self, query: str) -> List[str]:
        """Generates paraphrased and contradictory statements for semantic search."""
        # Simple for now, can be expanded with an LLM or more complex rules
        variations = [query]
        if "higher" in query:
            variations.append(query.replace("higher", "lower"))
        if "increase" in query:
            variations.append(query.replace("increase", "decrease"))
        if "indicated for" in query:
            variations.append(query.replace("indicated for", "not recommended for"))
        return variations

    def run(self, question: str, reference_contexts: List[str] = None) -> Dict[str, Any]:
        """Executes the full 3-stage retrieval and answer generation pipeline."""
        k = 10  # Retrieve more candidates initially

        # --- Stage 1: Parallel Augmented Retrieval ---
        keyword_queries = self._generate_keyword_queries(question)
        # Tokenize each keyword query for BM25
        tokenized_keyword_queries = [self._tokenize_and_stem(query) for query in keyword_queries]
        bm25_results_indices, _ = self.bm25_index.retrieve(tokenized_keyword_queries, k=k)
        
        semantic_statements = self._augment_semantic_statements(question)
        query_embeddings = self.embedding_model.encode(semantic_statements, convert_to_numpy=True).astype('float32')
        _, faiss_results_indices = self.faiss_index.search(query_embeddings, k)

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
            topic_counts = Counter(doc.get('topic_name', 'Unknown') for doc in candidate_docs)
            winning_topic = topic_counts.most_common(1)[0][0]

        # --- Stage 3: Focused BM25 Reranking ---
        # Filter for documents from the winning topic
        on_topic_docs = [doc for doc in self.documents if doc.get('topic_name') == winning_topic]
        on_topic_texts = [doc['text'] for doc in on_topic_docs]

        if not on_topic_docs:
            # Fallback to all documents if no docs for the winning topic
            on_topic_docs = self.documents
            on_topic_texts = self.document_texts
        
        # Create a temporary BM25 index for just the on-topic docs
        topic_bm25 = bm25s.BM25()
        topic_tokenized_corpus = [self._tokenize_and_stem(text) for text in on_topic_texts]
        topic_bm25.index(topic_tokenized_corpus)
        
        # Rerank on-topic documents using the original query for precision
        query_tokens = self._tokenize_and_stem(question)
        final_indices, _ = topic_bm25.retrieve([query_tokens], k=5)

        retrieved_docs = [on_topic_docs[i] for i in final_indices[0]]

        # --- Final Answer Generation ---
        context = "\n".join([doc['text'] for doc in retrieved_docs])
        llm_response = self.llm_client.classify_statement(question, context)

        answer = {
            "statement_is_true": llm_response[0],
            "statement_topic": self.llm_client._topic_name_to_number(winning_topic),
        }

        return {
            "answer": json.dumps(answer),
            "context": [doc['text'] for doc in retrieved_docs],
        }

