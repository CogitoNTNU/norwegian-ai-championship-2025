"""
Hybrid RAG template for Apple Silicon - optimized for emergency healthcare classification
Uses our custom hybrid RAG system with BM25s + FAISS + Qwen reranker
"""

import json
import asyncio
import sys
import os
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from hybrid.pipeline import run_with_contexts as hybrid_run
from llm_client import LocalLLMClient
class HybridRAGAppleSilicon:
    """Hybrid RAG system specialized for emergency healthcare using Apple Silicon optimization."""
    
    def __init__(self, llm_client: LocalLLMClient = None):
        self.llm_client = llm_client or LocalLLMClient()
        # The hybrid system handles its own LLM communication via Ollama
        print("✅ HybridRAGAppleSilicon initialized - using hybrid pipeline")
    
    def run(self, question: str, reference_contexts: List[str] = None) -> Dict[str, Any]:
        """
        Process a medical statement question using the hybrid RAG pipeline.
        
        Args:
            question: The medical statement to evaluate
            reference_contexts: Optional reference contexts (not used by hybrid system)
            
        Returns:
            Dict with 'answer' and 'context' keys
        """
        try:
            # Use the hybrid pipeline to process the statement
            # This will internally handle query generation, retrieval, reranking, and classification
            result = asyncio.run(hybrid_run(question))
            
            # Format answer as JSON string
            answer = {
                "statement_is_true": result.get("statement_is_true", 1),
                "statement_topic": result.get("statement_topic", 0),
            }
            
            retrieved_contexts = result.get("retrieved_contexts", [])
            
            return {
                "answer": json.dumps(answer),
                "context": retrieved_contexts
            }
            
        except Exception as e:
            print(f"Error in HybridRAGAppleSilicon.run: {e}")
            import traceback
            traceback.print_exc()
            # Return safe defaults
            return {
                "answer": json.dumps({"statement_is_true": 1, "statement_topic": 0}),
                "context": [],
            }
