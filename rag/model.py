import sys
import os
from typing import Tuple
import json

# Add rag-evaluation/templates to path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "rag-evaluation", "templates")
)

# Also add rag-evaluation directory for llm_client import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag-evaluation"))

from hybrid_rag_apple_silicon import HybridRAGAppleSilicon
from llm_client import LocalLLMClient

# Get configuration from environment or use defaults
LLM_MODEL = os.getenv("LLM_MODEL", "cogito:3b")

# Initialize the RAG system once at module level for efficiency
print(f"Initializing HybridRAGAppleSilicon system with {LLM_MODEL}...")
llm_client = LocalLLMClient(model_name=LLM_MODEL)
rag_system = HybridRAGAppleSilicon(llm_client=llm_client)
print("HybridRAGAppleSilicon system ready!")


### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###
def predict(statement: str) -> Tuple[int, int]:
    """
    Predict both binary classification (true/false) and topic classification for a medical statement.
    Uses the HybridRAGAppleSilicon system combining BioBERT semantic search with BM25 keyword search.

    Args:
        statement (str): The medical statement to classify

    Returns:
        Tuple[int, int]: (statement_is_true, statement_topic)
            - statement_is_true: 1 if true, 0 if false
            - statement_topic: topic ID from 0-114
    """
    try:
        # Use the RAG system to get predictions
        result = rag_system.run(statement)

        # The result['answer'] is a JSON string, so we need to parse it
        answer_data = json.loads(result["answer"])

        statement_is_true = answer_data.get("statement_is_true", 1)
        statement_topic = answer_data.get("statement_topic", 0)

        # Ensure values are in valid ranges
        statement_is_true = max(0, min(1, statement_is_true))
        statement_topic = max(0, min(114, statement_topic))

        return statement_is_true, statement_topic

    except Exception as e:
        print(f"Error in prediction: {e}")
        # Fallback to safe defaults
        return 1, 0
