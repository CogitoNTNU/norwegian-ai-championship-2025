"""
Query Rewrite RAG template.
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


class QueryRewriteRAG:
    """Query Rewrite RAG system for enhanced question processing."""
    
    def __init__(self, llm_client: LocalLLMClient = None):
        """Initialize QueryRewriteRAG with LocalLLMClient."""
        self.llm_client = llm_client or LocalLLMClient()
        self.embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    
    def _split_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
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
    
    def rewrite_query(self, question: str) -> str:
        """Rewrite the query for better retrieval."""
        prompt = f"""Rewrite this medical question to be more specific and detailed for better information retrieval:

Original: {question}
Rewritten:"""

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
                        "num_predict": 100,
                    },
                },
                timeout=15
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return response_data.get("response", question).strip()
            else:
                print(f"Error from Ollama server: {response.status_code}")
                return question

        except Exception as e:
            print(f"Error rewriting query: {e}")
            return question
    
    def retrieve_context(self, rewritten_query: str, reference_contexts: List[str], k: int = 5) -> List[str]:
        """Retrieve contexts using the rewritten query."""
        if not reference_contexts:
            return []
        
        try:
            # Create documents from reference contexts
            docs = []
            for context in reference_contexts:
                chunks = self._split_text(context)
                for chunk in chunks:
                    if chunk.strip():
                        docs.append(Document(page_content=chunk))
            
            if not docs:
                return reference_contexts[:k]
            
            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                collection_name="query-rewrite-temp"
            )
            
            # Use rewritten query for retrieval
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            retrieved_docs = retriever.invoke(rewritten_query)
            
            contexts = [doc.page_content for doc in retrieved_docs]
            return contexts
            
        except Exception as e:
            print(f"Error in Query Rewrite retrieval: {e}")
            return reference_contexts[:k]
    
    def run(self, question: str, reference_contexts: List[str] = None) -> Dict[str, Any]:
        """
        Process a question using query rewriting for better retrieval.
        
        Args:
            question: The question to answer
            reference_contexts: List of reference context strings
            
        Returns:
            Dict with 'answer' and 'context' keys
        """
        try:
            if not reference_contexts:
                reference_contexts = []
            
            # Rewrite the query for better retrieval
            rewritten_query = self.rewrite_query(question)
            
            # Retrieve contexts using rewritten query
            retrieved_contexts = self.retrieve_context(rewritten_query, reference_contexts, k=5)
            
            # Combine contexts for LLM
            combined_context = "\n".join(retrieved_contexts[:3])  # Limit to top 3
            
            # Use LLM to classify the original statement
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
            print(f"Error in QueryRewriteRAG.run: {e}")
            # Return safe defaults
            return {
                "answer": json.dumps({"statement_is_true": 1, "statement_topic": 0}),
                "context": [],
            }
