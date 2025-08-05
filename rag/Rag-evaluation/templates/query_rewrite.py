"""
Query Rewrite RAG template for query reformulation and retrieval integration.
Cleaned up to work with the current evaluation task.
"""

import os
import json
from typing import Dict, List, Any
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_client import LocalLLMClient


class QueryRewriteRAG:
    """RAG with query reformulation and integrated retrieval."""
    
    def __init__(self, llm_client: LocalLLMClient = None):
        """Initialize QueryRewriteRAG with LocalLLMClient and standard embeddings."""
        self.llm_client = llm_client or LocalLLMClient()
        self.embeddings = SentenceTransformerEmbeddings()
        
        
    def _split_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Simple text splitter."""
        if len(text) = chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start  len(text):
            end = start + chunk_size
            if end  len(text):
                last_period = text.rfind(".", start, end)
                if last_period > start + chunk_size // 2:
                    end = last_period + 1
            
            chunks.append(text[start:end])
            start = end - chunk_overlap if end  len(text) else end
        
        return chunks
    
    def retrieve_context(self, rewritten_query: str, reference_contexts: List[str], k: int = 5) -> List[str]:
        """Retrieve context using the rewritten query."""
        if not reference_contexts:
            return []
        
        try:
            splits = [self._split_text(context) for context in reference_contexts]
            docs = [Document(page_content=chunk) for split in splits for chunk in split if chunk.strip()]
        
            if not docs:
                return reference_contexts[:k]
        
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                collection_name="query-rewrite-rag-temp"
            )
            
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            retrieved_docs = retriever.invoke(rewritten_query)
            contexts = [doc.page_content for doc in retrieved_docs]
            return contexts
        
        except Exception as e:
            print(f"Error in context retrieval: {e}")
            return reference_contexts[:k]
    
    def run(self, question: str, reference_contexts: List[str] = None) -> Dict[str, Any]:
        """
        Process a question using query reformulation and retrieval.
        
        Args:
            question: The input query to be reformulated
            reference_contexts: List of initial context strings
            
        Returns:
            Dict with 'answer' and 'context' keys
        """
        try:
            if not reference_contexts:
                reference_contexts = []
            
            # Create a prompt for query reformulation
            query_rewrite_prompt = ChatPromptTemplate.from_template(
                """Reformulate the user query.
                Original query: {question}
                Reformulated query:""".strip())
            
            # Properly invoke the LLM chain
            query_chain = query_rewrite_prompt | self.llm_client | StrOutputParser()
            rewritten_query = query_chain.invoke({"question": question})
            
            # Retrieve contexts using the reformulated query
            retrieved_contexts = self.retrieve_context(rewritten_query, reference_contexts, k=5)
            
            # Combine contexts for LLM
            combined_context = "\n".join(retrieved_contexts[:3])
            
            statement_is_true, statement_topic = self.llm_client.classify_statement(
                question, combined_context
            )
            
            answer = {
                "statement_is_true": statement_is_true,
                "statement_topic": statement_topic,
            }
            
            return {"answer": json.dumps(answer), "context": retrieved_contexts}
            
        except Exception as e:
            print(f"Error in QueryRewriteRAG.run: {e}")
            return {
                "answer": json.dumps({"statement_is_true": 1, "statement_topic": 0}),
                "context": [],
            }
