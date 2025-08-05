"""
RAG template using the EmbeddingsRAGPipeline from rag-pipeline.
This template integrates the advanced embedding-based RAG system into the evaluation framework.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# Add paths to import from rag-pipeline
current_dir = Path(__file__).parent
rag_root = current_dir.parent.parent
rag_pipeline_dir = rag_root / "rag-pipeline"
sys.path.insert(0, str(rag_pipeline_dir))

try:
    from rag_pipeline_embeddings import EmbeddingsRAGPipeline
except ImportError as e:
    print(f"Error importing from rag-pipeline: {e}")
    print(f"Make sure rag-pipeline is in the correct location: {rag_pipeline_dir}")
    raise

# Import evaluation framework LLM client for fallback
try:
    from llm_client import LocalLLMClient
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from common.client import LocalLLMClient


class LLMClientWrapper:
    """Wrapper to directly use the evaluation framework LLM client."""

    def __init__(self, eval_llm_client: LocalLLMClient):
        self.eval_client = eval_llm_client

    def classify_statement(self, statement: str, context: str):
        """Directly call classify_statement."""
        return self.eval_client.classify_statement(statement, context)
    
    def ensure_model_available(self):
        """Ensure the model is available."""
        return self.eval_client.ensure_model_available()


class EmbeddingsRAG:
    """RAG system using configurable embedding models from rag-pipeline."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "qwen3:8b",
        top_k_retrieval: int = 5,
        retrieval_strategy: str = "default",
        llm_client: LocalLLMClient = None
    ):
        """
        Initialize EmbeddingsRAG with configurable models.
        
        Args:
            embedding_model: Model name from embeddings registry
            llm_model: Local LLM model name
            top_k_retrieval: Number of relevant chunks to retrieve
            retrieval_strategy: Strategy to use ("default", "hyde", "hybrid")
            llm_client: Optional external LLM client (for evaluation compatibility)
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.top_k = top_k_retrieval
        self.retrieval_strategy = retrieval_strategy
        
        # Initialize RAG pipeline
        try:
            self.rag_pipeline = EmbeddingsRAGPipeline(
                embedding_model=embedding_model,
                llm_model=llm_model,
                top_k_retrieval=top_k_retrieval,
                retrieval_strategy=retrieval_strategy
            )
            
            # If an external LLM client is provided, wrap it for compatibility
            if llm_client:
                self.rag_pipeline.llm_client = LLMClientWrapper(llm_client)
            
            self._setup_pipeline()
            # Ensure the document index is built before running
            if not self.rag_pipeline.document_store.is_index_built():
                print("Building document index...")
                self.rag_pipeline.document_store.build_index()
            self.pipeline_ready = True
        except Exception as e:
            print(f"Error initializing RAG pipeline: {e}")
            self.pipeline_ready = False
            # Fallback to evaluation framework LLM client
            self.llm_client = llm_client or LocalLLMClient()

    def _setup_pipeline(self):
        """Set up the RAG pipeline with medical documents."""
        try:
            # Find paths to data directories
            topics_dir = self._find_topics_directory()
            topics_json = self._find_topics_json()
            
            if not topics_dir or not topics_json:
                print("Warning: Could not find topics directory or topics.json")
                self.pipeline_ready = False
                return
            
            print(f"Setting up RAG pipeline with {self.embedding_model}...")
            self.rag_pipeline.setup(topics_dir, topics_json)
            print("RAG pipeline setup complete!")
            
        except Exception as e:
            print(f"Error setting up RAG pipeline: {e}")
            self.pipeline_ready = False

    def _find_topics_directory(self) -> str:
        """Find the topics directory."""
        possible_paths = [
            "data/raw/topics",  # Direct path when running from rag root
            "../data/raw/topics",
            "../../data/raw/topics",
            str(Path(__file__).parent.parent.parent / "data" / "raw" / "topics"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def _find_topics_json(self) -> str:
        """Find the topics.json file."""
        possible_paths = [
            "data/topics.json",  # Direct path when running from rag root
            "../data/topics.json", 
            "../../data/topics.json",
            str(Path(__file__).parent.parent.parent / "data" / "topics.json"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def retrieve_context(self, query: str, k: int = None) -> List[str]:
        """
        Retrieve relevant medical contexts for the query.
        
        Args:
            query: Search query
            k: Number of results to return (uses self.top_k if None)
            
        Returns:
            List of retrieved context strings
        """
        if not self.pipeline_ready:
            return []
            
        k = k or self.top_k
        
        try:
            # Use the retrieval strategy to get relevant chunks
            relevant_chunks = self.rag_pipeline.retrieval_strategy.retrieve(
                query, self.rag_pipeline.document_store, k=k
            )
            
            # Extract context strings from chunks
            contexts = []
            for chunk_data in relevant_chunks:
                context = chunk_data["chunk"]
                metadata = chunk_data["metadata"]
                
                # Add topic information for better context
                if "topic_name" in metadata:
                    context = f"[{metadata['topic_name']}] {context}"
                
                contexts.append(context)
            
            return contexts
            
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []

    def run(self, question: str, reference_contexts: List[str] = None) -> Dict[str, Any]:
        """
        Process a medical statement question using the embeddings RAG pipeline.
        
        Args:
            question: The medical statement to evaluate
            reference_contexts: Optional reference contexts (not used in this implementation)
            
        Returns:
            Dict with 'answer' and 'context' keys
        """
        try:
            if self.pipeline_ready:
                # Use the full RAG pipeline for prediction
                statement_is_true, statement_topic = self.rag_pipeline.predict(question)
                
                # Also get the retrieved contexts for evaluation
                retrieved_contexts = self.retrieve_context(question)
                
            else:
                # Fallback: use evaluation framework LLM client with empty context
                print("Pipeline not ready, using fallback...")
                statement_is_true, statement_topic = self.llm_client.classify_statement(
                    question, ""
                )
                retrieved_contexts = []

            # Format answer for evaluation
            answer = {
                "statement_is_true": statement_is_true,
                "statement_topic": statement_topic,
            }

            return {"answer": json.dumps(answer), "context": retrieved_contexts}

        except Exception as e:
            print(f"Error in EmbeddingsRAG.run: {e}")
            # Return safe defaults
            return {
                "answer": json.dumps({"statement_is_true": 1, "statement_topic": 0}),
                "context": [],
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        if self.pipeline_ready:
            return self.rag_pipeline.get_model_info()
        else:
            return {
                "embedding_model": self.embedding_model,
                "llm_model": self.llm_model,
                "top_k": self.top_k,
                "retrieval_strategy": self.retrieval_strategy,
                "pipeline_ready": False
            }

    def evaluate_on_training_data(
        self, max_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate pipeline on training data for debugging.
        
        Args:
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.pipeline_ready:
            return {"error": "Pipeline not ready"}
            
        try:
            # Find training data directories
            training_statements_dir = self._find_training_statements()
            training_answers_dir = self._find_training_answers()
            
            if not training_statements_dir or not training_answers_dir:
                return {"error": "Training data directories not found"}
                
            return self.rag_pipeline.evaluate_on_training_data(
                training_statements_dir, training_answers_dir, max_samples
            )
            
        except Exception as e:
            return {"error": f"Evaluation failed: {e}"}

    def _find_training_statements(self) -> str:
        """Find training statements directory."""
        possible_paths = [
            "data/processed/combined/statements",
            "../data/processed/combined/statements", 
            "../../data/processed/combined/statements",
            str(Path(__file__).parent.parent.parent / "data" / "processed" / "combined" / "statements"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def _find_training_answers(self) -> str:
        """Find training answers directory."""
        possible_paths = [
            "data/processed/combined/answers",
            "../data/processed/combined/answers",
            "../../data/processed/combined/answers", 
            str(Path(__file__).parent.parent.parent / "data" / "processed" / "combined" / "answers"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None


# Factory functions for different embedding models
def create_minilm_rag(**kwargs) -> EmbeddingsRAG:
    """Create RAG with all-MiniLM-L6-v2 embeddings."""
    return EmbeddingsRAG(embedding_model="all-MiniLM-L6-v2", **kwargs)

def create_e5_small_rag(**kwargs) -> EmbeddingsRAG:
    """Create RAG with E5-small-v2 embeddings."""
    return EmbeddingsRAG(embedding_model="e5-small-v2", **kwargs)

def create_e5_base_rag(**kwargs) -> EmbeddingsRAG:
    """Create RAG with E5-base-v2 embeddings."""
    return EmbeddingsRAG(embedding_model="e5-base-v2", **kwargs)

def create_hyde_rag(**kwargs) -> EmbeddingsRAG:
    """Create RAG with HyDE retrieval strategy."""
    return EmbeddingsRAG(retrieval_strategy="hyde", **kwargs)

def create_hybrid_rag(**kwargs) -> EmbeddingsRAG:
    """Create RAG with hybrid retrieval strategy."""
    return EmbeddingsRAG(retrieval_strategy="hybrid", **kwargs)
