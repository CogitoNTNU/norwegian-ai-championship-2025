"""
Query Expansion with RRF (Reciprocal Rank Fusion) RAG template.
Simplified for evaluation framework compatibility.
"""

import os
import json
import requests
from typing import Dict, List, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

try:
    from ..llm_client import LocalLLMClient
except ImportError:
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from llm_client import LocalLLMClient


class QueryExpansionRRF:
    """Query Expansion RAG with simplified RRF (Reciprocal Rank Fusion)."""

    def __init__(self, llm_client: LocalLLMClient = None):
        """Initialize QueryExpansionRRF with LocalLLMClient."""
        self.llm_client = llm_client or LocalLLMClient()
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n".join(doc.page_content for doc in docs)

    def _rrf(self, results: List[List[str]]) -> List[str]:
        fused_scores = {}
        k = 60
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = doc  # Use document as a string representation
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            doc
            for doc, score in sorted(
                fused_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]
        return reranked_results

    def enhance_query(self, question: str) -> List[str]:
        """Generate expanded queries based on the original question."""
        prompt = f"""Given this question, generate expanded search queries:

Question: {question}
Expanded Queries:"""

        try:
            response = requests.post(
                f"{self.llm_client.ollama_url}/api/generate",
                json={
                    "model": self.llm_client.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.8,
                        "num_predict": 200,
                    },
                },
                timeout=15,
            )

            if response.status_code == 200:
                response_data = response.json()
                return response_data.get("response", question).strip().split("\n")
            else:
                print(f"Error from Ollama server: {response.status_code}")
                return [question]

        except Exception as e:
            print(f"Error generating expanded queries: {e}")
            return [question]

    def run(
        self, question: str, reference_contexts: List[str] = None
    ) -> Dict[str, Any]:
        """
        Process a question using query expansion and RRF.

        Args:
            question: The question to answer
            reference_contexts: List of reference context strings

        Returns:
            Dict with 'answer' and 'context' keys
        """
        try:
            if not reference_contexts:
                reference_contexts = []

            # Generate expanded queries
            expanded_queries = self.enhance_query(question)

            # Use expanded queries to retrieve contexts
            all_docs = []
            for query in expanded_queries:
                docs = [
                    Document(page_content=content) for content in reference_contexts
                ]
                vectorstore = Chroma.from_documents(
                    documents=docs, embedding=self.embeddings
                )
                retriever = vectorstore.as_retriever()
                retrieved_docs = retriever.invoke(query)
                all_docs.append([doc.page_content for doc in retrieved_docs])

            # Fuse results
            fused_contexts = self._rrf(all_docs)

            # Combine top k contexts
            combined_context = "\n".join(fused_contexts[:5])

            # Use LLM to generate final answer using enriched context
            statement_is_true, statement_topic = self.llm_client.classify_statement(
                question, combined_context
            )

            # Format answer for evaluation
            answer = {
                "statement_is_true": statement_is_true,
                "statement_topic": statement_topic,
            }

            return {"answer": json.dumps(answer), "context": fused_contexts}

        except Exception as e:
            print(f"Error in QueryExpansionRRF.run: {e}")
            # Return safe defaults
            return {
                "answer": json.dumps({"statement_is_true": 1, "statement_topic": 0}),
                "context": [],
            }
