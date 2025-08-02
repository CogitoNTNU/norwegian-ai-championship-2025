import json
import sys
import os
from typing import Tuple

# Add the Rag-evaluation/src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Rag-evaluation/src"))

from templates.healthcare_rag import HealthcareRAG
from llm_client import LocalLLMClient

# Initialize the RAG system once at module level for efficiency
print("Initializing BM25s Healthcare RAG system...")
llm_client = LocalLLMClient()
rag_system = HealthcareRAG(llm_client)
print("Healthcare RAG system ready!")


### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###
def predict(statement: str) -> Tuple[int, int]:
    """
    Predict both binary classification (true/false) and topic classification for a medical statement.
    Uses the optimized BM25s-powered Healthcare RAG system.

    Args:
        statement (str): The medical statement to classify

    Returns:
        Tuple[int, int]: (statement_is_true, statement_topic)
            - statement_is_true: 1 if true, 0 if false
            - statement_topic: topic ID from 0-114
    """
    try:
        # Use the RAG system to get classification
        result = rag_system.run(statement)

        # Parse the JSON answer
        answer_json = json.loads(result["answer"])
        statement_is_true = int(answer_json.get("statement_is_true", 1))
        statement_topic = int(answer_json.get("statement_topic", 0))

        # Ensure values are in valid ranges
        statement_is_true = max(0, min(1, statement_is_true))
        statement_topic = max(0, min(114, statement_topic))

        return statement_is_true, statement_topic

    except Exception as e:
        print(f"Error in prediction: {e}")
        # Fallback to safe defaults
        return 1, 0


def match_topic(statement: str) -> int:
    """
    Fallback simple keyword matching to find the best topic match.
    This is used as a backup if the main RAG system fails.
    """
    try:
        # Try to load topics mapping from different possible locations
        topics_paths = [
            "data/topics.json",
            "../data/topics.json",
            "Rag-evaluation/data/topics.json",
            os.path.join(os.path.dirname(__file__), "..", "data", "topics.json"),
        ]

        topics = {}
        for path in topics_paths:
            if os.path.exists(path):
                with open(path, "r") as f:
                    topics = json.load(f)
                break

        if not topics:
            print("Warning: Could not load topics.json, using default topic 0")
            return 0

        statement_lower = statement.lower()
        best_topic = 0
        max_matches = 0

        for topic_name, topic_id in topics.items():
            # Extract keywords from topic name
            keywords = (
                topic_name.lower()
                .replace("_", " ")
                .replace("(", "")
                .replace(")", "")
                .split()
            )

            # Count keyword matches in statement
            matches = sum(1 for keyword in keywords if keyword in statement_lower)

            if matches > max_matches:
                max_matches = matches
                best_topic = topic_id

        return best_topic

    except Exception as e:
        print(f"Error in topic matching: {e}")
        return 0
