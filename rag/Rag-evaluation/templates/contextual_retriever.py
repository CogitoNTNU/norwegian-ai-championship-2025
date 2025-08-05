"""
Contextual Retriever RAG template - simplified for evaluation framework compatibility.
"""

import os
import json
from typing import Dict, List, Any
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

try:
    from ..llm_client import LocalLLMClient
except ImportError:
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from llm_client import LocalLLMClient


class ContextualRetrieverRAG:
    """Simplified Contextual RAG system using enhanced context for retrieval."""

    def __init__(self, llm_client: LocalLLMClient = None):
        """Initialize ContextualRetrieverRAG with LocalLLMClient."""
        self.llm_client = llm_client or LocalLLMClient()
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

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
                last_period = text.rfind(".", start, end)
                if last_period > start + chunk_size // 2:
                    end = last_period + 1

            chunks.append(text[start:end])
            start = end - chunk_overlap if end < len(text) else end

        return chunks

    def enhance_context(self, text: str, question: str) -> str:
        """Add contextual information to improve retrieval."""
        # Simplified contextual enhancement - just add question-relevant prefix
        medical_keywords = [
            "medical",
            "patient",
            "treatment",
            "diagnosis",
            "symptoms",
            "emergency",
        ]
        question_lower = question.lower()

        context_prefix = "Medical context: "
        for keyword in medical_keywords:
            if keyword in question_lower:
                context_prefix += f"This information relates to {keyword}. "
                break

        return context_prefix + text

    def retrieve_context(
        self, question: str, reference_contexts: List[str], k: int = 5
    ) -> List[str]:
        """Retrieve contexts with contextual enhancement."""
        if not reference_contexts:
            return []

        try:
            # Create documents with enhanced context
            docs = []
            for context in reference_contexts:
                chunks = self._split_text(context)
                for chunk in chunks:
                    if chunk.strip():
                        enhanced_chunk = self.enhance_context(chunk, question)
                        docs.append(Document(page_content=enhanced_chunk))

            if not docs:
                return reference_contexts[:k]

            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                collection_name="contextual-retriever-temp",
            )

            # Retrieve similar documents
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            retrieved_docs = retriever.invoke(question)

            # Extract content (remove the contextual prefix for cleaner output)
            contexts = []
            for doc in retrieved_docs:
                content = doc.page_content
                if content.startswith("Medical context:"):
                    # Remove the contextual prefix
                    content = content.split("Medical context:", 1)[1].strip()
                    if content.startswith("This information relates to"):
                        # Remove the enhanced prefix
                        parts = content.split(". ", 1)
                        if len(parts) > 1:
                            content = parts[1]
                contexts.append(content)

            return contexts

        except Exception as e:
            print(f"Error in Contextual retrieval: {e}")
            return reference_contexts[:k]

    def run(
        self, question: str, reference_contexts: List[str] = None
    ) -> Dict[str, Any]:
        """
        Process a question using contextual retrieval.

        Args:
            question: The question to answer
            reference_contexts: List of reference context strings

        Returns:
            Dict with 'answer' and 'context' keys
        """
        try:
            if not reference_contexts:
                reference_contexts = []

            # Retrieve contexts with contextual enhancement
            retrieved_contexts = self.retrieve_context(
                question, reference_contexts, k=5
            )

            # Combine contexts for LLM
            combined_context = "\n".join(retrieved_contexts[:3])  # Limit to top 3

            # Use LLM to classify the statement
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
            print(f"Error in ContextualRetrieverRAG.run: {e}")
            # Return safe defaults
            return {
                "answer": json.dumps({"statement_is_true": 1, "statement_topic": 0}),
                "context": [],
            }
