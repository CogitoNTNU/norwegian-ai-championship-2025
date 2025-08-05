"""
PocketFlow RAG Template for evaluation integration - Simplified version
"""

import os
import json
import subprocess
import tempfile
from typing import Dict, List, Any

try:
    from ..llm_client import LocalLLMClient
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from llm_client import LocalLLMClient


class PocketFlowRAG:
    def __init__(self, llm_client=None, llm=None, embeddings=None):
        """
        Initialize PocketFlow RAG template
        Args:
            llm_client: LocalLLMClient from evaluation framework (optional)
            llm: RAGAS-provided LLM (optional, we'll ignore this)
            embeddings: RAGAS-provided embeddings (optional, we'll ignore this)
        """
        self.llm_client = llm_client
        self.llm = llm
        self.embeddings = embeddings
        self.custom_rag_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "RAG-chat-frontend-backend",
                "custom-rag",
            )
        )

    def run(self, question, context):
        """
        Run PocketFlow-style RAG using local LLM client
        Args:
            question: User query string
            context: List of reference context strings (from testset)
        Returns:
            dict with 'answer' and 'context' keys for RAGAS evaluation
        """
        try:
            # Use local LLM client if available, otherwise fallback to simple response
            if self.llm_client:
                # Combine contexts for LLM
                combined_context = "\n".join(context[:3]) if context else ""
                
                # Use LLM to classify the statement (following the pattern of other templates)
                statement_is_true, statement_topic = self.llm_client.classify_statement(
                    question, combined_context
                )
                
                # Format answer for evaluation (similar to simple_rag)
                answer = {
                    "statement_is_true": statement_is_true,
                    "statement_topic": statement_topic,
                }
                
                return {
                    "answer": json.dumps(answer),
                    "context": context[:3] if context else ["No context available"],
                }
            else:
                # Fallback response when no LLM client is available
                return {
                    "answer": f"I understand you're asking about: {question}. However, I would need access to the PocketFlow RAG system to provide a detailed answer.",
                    "context": context[:3] if context else ["No context available"],
                }
                
        except Exception as e:
            print(f"Error in PocketFlow RAG adapter: {e}")
            import traceback
            traceback.print_exc()
            
            # Return fallback response
            return {
                "answer": f"Error generating response: {str(e)}",
                "context": context[:3] if context else [],
            }
