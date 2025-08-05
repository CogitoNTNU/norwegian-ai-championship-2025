"""
Hybrid RAG template combining BM25s and semantic similarity retrieval.
Cleaned up to work with the current evaluation task.
"""

import os
import json
import re
from typing import Dict, List, Any
import bm25s
import Stemmer
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

try:
    from ..llm_client import LocalLLMClient
except ImportError:
    # Fallback for direct execution
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from llm_client import LocalLLMClient


class HybridRAG:
    """Hybrid RAG system combining BM25s and semantic similarity retrieval."""

    def __init__(self, llm_client: LocalLLMClient = None):
        """Initialize HybridRAG with LocalLLMClient and hybrid retrieval."""
        self.llm_client = llm_client or LocalLLMClient()
        # Use LangChain embedding wrapper
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.stemmer = Stemmer.Stemmer("english")

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
        tokens = re.findall(r"\b\w+\b", text.lower())
        stemmed_tokens = self.stemmer.stemWords(tokens)
        return stemmed_tokens

    def retrieve_context(
        self, query: str, reference_contexts: List[str], k: int = 5
    ) -> List[str]:
        """Retrieve relevant contexts using hybrid BM25s + semantic similarity."""
        if not reference_contexts:
            return []

        try:
            # Prepare documents
            docs = []
            doc_texts = []

            for context in reference_contexts:
                chunks = self._split_text(context)
                for chunk in chunks:
                    if chunk.strip():
                        docs.append(Document(page_content=chunk))
                        doc_texts.append(chunk)

            if not doc_texts:
                return reference_contexts[:k]

            # BM25s retrieval
            bm25_scores = []
            try:
                tokenized_corpus = [self._tokenize_and_stem(doc) for doc in doc_texts]
                bm25_index = bm25s.BM25()
                bm25_index.index(tokenized_corpus)

                query_tokens = self._tokenize_and_stem(query)
                bm25_results = bm25_index.retrieve([query_tokens], k=len(doc_texts))

                # Get BM25 scores
                bm25_scores = bm25_results.scores[0]

            except Exception as e:
                print(f"BM25s error: {e}")
                # Fallback to equal scores for all documents
                bm25_scores = [1.0] * len(doc_texts)

            # Semantic similarity retrieval
            semantic_scores = []
            try:
                vectorstore = Chroma.from_documents(
                    documents=docs,
                    embedding=self.embeddings,
                    collection_name="hybrid-rag-temp",
                )

                retriever = vectorstore.as_retriever(search_kwargs={"k": len(docs)})
                semantic_results = retriever.invoke(query)

                # Create semantic scores (approximate based on order)
                semantic_scores = [1.0 / (i + 1) for i in range(len(semantic_results))]

            except Exception as e:
                print(f"Semantic retrieval error: {e}")
                semantic_scores = [0.5] * len(doc_texts)

            # Combine scores (weighted average)
            bm25_weight = 0.6
            semantic_weight = 0.4

            combined_scores = []
            for i in range(len(doc_texts)):
                bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0.0
                semantic_score = semantic_scores[i] if i < len(semantic_scores) else 0.0

                combined_score = (
                    bm25_weight * bm25_score + semantic_weight * semantic_score
                )
                combined_scores.append((i, combined_score))

            # Sort by combined score and get top k
            combined_scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in combined_scores[:k]]

            # Return top contexts
            contexts = [doc_texts[i] for i in top_indices]
            return contexts

        except Exception as e:
            print(f"Error in hybrid retrieval: {e}")
            return reference_contexts[:k]

    def run(
        self, question: str, reference_contexts: List[str] = None
    ) -> Dict[str, Any]:
        """
        Process a question using hybrid BM25s + semantic retrieval.

        Args:
            question: The question to answer
            reference_contexts: List of reference context strings

        Returns:
            Dict with 'answer' and 'context' keys
        """
        try:
            if not reference_contexts:
                reference_contexts = []

            # Retrieve relevant contexts using hybrid approach
            retrieved_contexts = self.retrieve_context(
                question, reference_contexts, k=5
            )

            # Combine contexts for LLM
            combined_context = "\n".join(retrieved_contexts[:3])  # Limit to top 3

            # Use LLM to generate answer
            statement_is_true, statement_topic = self.llm_client.classify_statement(
                question, combined_context
            )

            # Format answer for evaluation
            answer = {
                "statement_is_true": statement_is_true,
                "statement_topic": statement_topic,
            }

            return {"answer": json.dumps(answer), "context": retrieved_contexts}

        except Exception as e:
            print(f"Error in HybridRAG.run: {e}")
            # Return safe defaults
            return {
                "answer": json.dumps({"statement_is_true": 1, "statement_topic": 0}),
                "context": [],
            }
