"""
Simple RAG template using semantic similarity retrieval.
Cleaned up to work with the current evaluation task.
"""

import os
import json
from typing import Dict, List, Any
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

try:
    from ..llm_client import LocalLLMClient
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from llm_client import LocalLLMClient


class SimpleRAG:
    """Simple RAG system using semantic similarity retrieval."""
    
    def __init__(self, llm_client: LocalLLMClient = None):
        """Initialize SimpleRAG with LocalLLMClient and standard embeddings."""
        self.llm_client = llm_client or LocalLLMClient()
        # Use a standard embedding model (sentence-transformers) with LangChain wrapper
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
                # Try to split at sentence boundary
                last_period = text.rfind(".", start, end)
                if last_period > start + chunk_size // 2:
                    end = last_period + 1
            
            chunks.append(text[start:end])
            start = end - chunk_overlap if end < len(text) else end
        
        return chunks
    
    def retrieve_context(self, query: str, reference_contexts: List[str], k: int = 5) -> List[str]:
        """Retrieve relevant contexts using semantic similarity."""
        if not reference_contexts:
            return []
        
        try:
            # Create documents from reference contexts
            docs = []
            for context in reference_contexts:
                # Split each context into chunks
                chunks = self._split_text(context)
                for chunk in chunks:
                    if chunk.strip():
                        docs.append(Document(page_content=chunk))
            
            if not docs:
                return reference_contexts[:k]  # Fallback to original contexts
            
            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                collection_name="simple-rag-temp"
            )
            
            # Retrieve similar documents
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            retrieved_docs = retriever.invoke(query)
            
            # Extract content
            contexts = [doc.page_content for doc in retrieved_docs]
            return contexts
            
        except Exception as e:
            print(f"Error in retrieval: {e}")
            # Fallback to first k reference contexts
            return reference_contexts[:k]
    
    def run(self, question: str, reference_contexts: List[str] = None) -> Dict[str, Any]:
        """
        Process a question using simple semantic similarity retrieval.
        
        Args:
            question: The question to answer
            reference_contexts: List of reference context strings
            
        Returns:
            Dict with 'answer' and 'context' keys
        """
        try:
            if not reference_contexts:
                reference_contexts = []
            
            # Retrieve relevant contexts
            retrieved_contexts = self.retrieve_context(question, reference_contexts, k=5)
            
            # Combine contexts for LLM
            combined_context = "\n".join(retrieved_contexts[:3])  # Limit to top 3
            
            # Use LLM to generate answer (following healthcare_rag pattern)
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
            print(f"Error in SimpleRAG.run: {e}")
            # Return safe defaults
            return {
                "answer": json.dumps({"statement_is_true": 1, "statement_topic": 0}),
                "context": [],
            }
