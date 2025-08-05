"""
Hybrid Retrieval Only Template - BM25s + FAISS without LLM generation
Tests pure retrieval capability for component classification
"""

import json
from typing import List, Dict, Any
from langchain_community.embeddings import SentenceTransformerEmbeddings
from llm_client import LocalLLMClient


class HybridRetrievalOnly:
    def __init__(self, llm_client: LocalLLMClient):
        """
        Initialize Hybrid Retrieval Only template with standard LLM client and embeddings
        """
        self.llm_client = llm_client
        self.embeddings = SentenceTransformerEmbeddings()

    def run(self, question: str, reference_contexts: List[str] = None) -> Dict[str, Any]:
        """
        Run hybrid retrieval only - Update to use LLM client for classification
        Args:
            question: User query string
            reference_contexts: List of reference context strings
        Returns:
            Dict with 'answer' and 'context' keys
        """
        if not reference_contexts:
            reference_contexts = []
        
        # Combine contexts for classification
        combined_context = "\n".join(reference_contexts[:3]) if reference_contexts else ""

        # Obtain classification
        statement_is_true, statement_topic = self.llm_client.classify_statement(
            question, combined_context
        )

        answer = {
            "statement_is_true": statement_is_true,
            "statement_topic": statement_topic,
        }

        return {
            "answer": json.dumps(answer),
            "context": reference_contexts[:3] if reference_contexts else ["No context available"],
        }
