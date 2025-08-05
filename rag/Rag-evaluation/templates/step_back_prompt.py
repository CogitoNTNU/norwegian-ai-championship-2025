"""
Step Back Prompt RAG template for context expansion.
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


class StepBackPromptRAG:
    """RAG for context expansion using step-back query prompts."""
    
    def __init__(self, llm_client: LocalLLMClient = None):
        """Initialize StepBackPromptRAG."""
        self.llm_client = llm_client or LocalLLMClient()
        self.embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

    def generate_step_back_query(self, question: str) -> str:
        """Generate a more general query to get broader context."""
        prompt = f"""Generate a more general query from this medical question:
        
Question: {question}
General Query:"""

        try:
            response = requests.post(
                f"{self.llm_client.ollama_url}/api/generate",
                json={
                    "model": self.llm_client.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.5,
                        "top_p": 0.9,
                        "num_predict": 50,
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
            print(f"Error generating step-back query: {e}")
            return question
    
    def retrieve_context(self, step_back_query: str, reference_contexts: List[str], k: int = 5) -> List[str]:
        """Retrieve contexts based on the step-back query."""
        if not reference_contexts:
            return []
        
        try:
            # Create documents from reference contexts
            docs = []
            for context in reference_contexts:
                docs.append(Document(page_content=context))
            
            if not docs:
                return reference_contexts[:k]
            
            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                collection_name="step-back-prompt-temp"
            )
            
            # Use step-back query for retrieval
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            retrieved_docs = retriever.invoke(step_back_query)
            
            contexts = [doc.page_content for doc in retrieved_docs]
            return contexts
            
        except Exception as e:
            print(f"Error in StepBack Retrieval: {e}")
            return reference_contexts[:k]

    def run(self, question: str, reference_contexts: List[str] = None) -> Dict[str, Any]:
        """
        Process a question using step-back query prompts.
        """
        try:
            if not reference_contexts:
                reference_contexts = []

            # Generate step-back query
            step_back_query = self.generate_step_back_query(question)

            # Retrieve context using step-back query
            retrieved_contexts = self.retrieve_context(step_back_query, reference_contexts, k=5)

            # Combine contexts for LLM
            combined_context = "\n".join(retrieved_contexts[:3])

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
            print(f"Error in StepBackPromptRAG.run: {e}")
            # Return safe defaults
            return {
                "answer": json.dumps({"statement_is_true": 1, "statement_topic": 0}),
                "context": [],
            }
