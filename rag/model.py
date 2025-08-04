import sys
import os
from typing import Tuple
from pathlib import Path

# Add rag-pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag-pipeline"))

from rag_pipeline_embeddings import EmbeddingsRAGPipeline

# Get configuration from environment or use defaults
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "pubmedbert-base-embeddings")
LLM_MODEL = os.getenv("LLM_MODEL", "cogito:3b")
RETRIEVAL_STRATEGY = os.getenv("RETRIEVAL_STRATEGY", "default")

# Initialize the RAG system once at module level for efficiency
print(f"Initializing Embeddings RAG system with {EMBEDDING_MODEL} and {LLM_MODEL}...")
print(f"Using retrieval strategy: {RETRIEVAL_STRATEGY}")
rag_pipeline = EmbeddingsRAGPipeline(
    embedding_model=EMBEDDING_MODEL,
    llm_model=LLM_MODEL,
    top_k_retrieval=5,
    retrieval_strategy=RETRIEVAL_STRATEGY,
)

# Setup with data
rag_dir = Path(__file__).parent
topics_dir = rag_dir / "data" / "topics"
topics_json = rag_dir / "data" / "topics.json"

rag_pipeline.setup(str(topics_dir), str(topics_json))
print("Embeddings RAG system ready!")


### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###
def predict(statement: str) -> Tuple[int, int]:
    """
    Predict both binary classification (true/false) and topic classification for a medical statement.
    Uses the embeddings-based RAG system with FAISS vector search.

    Args:
        statement (str): The medical statement to classify

    Returns:
        Tuple[int, int]: (statement_is_true, statement_topic)
            - statement_is_true: 1 if true, 0 if false
            - statement_topic: topic ID from 0-114
    """
    try:
        print(f"Starting prediction for: {statement[:50]}...")

        # Use the RAG pipeline to get predictions
        statement_is_true, statement_topic = rag_pipeline.predict(statement)

        print(f"Got prediction: is_true={statement_is_true}, topic={statement_topic}")

        # Ensure values are in valid ranges
        statement_is_true = max(0, min(1, statement_is_true))
        statement_topic = max(0, min(114, statement_topic))

        return statement_is_true, statement_topic

    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback

        traceback.print_exc()
        # Fallback to safe defaults
        return 1, 0
