from typing import Tuple
import sys
from pathlib import Path

# Add current directory to path so we can import our modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import after path modification to avoid E402  # noqa: E402
from rag_pipeline import RAGPipeline  # noqa: E402


class MedicalStatementClassifier:
    """Main classifier that integrates with the competition framework."""

    def __init__(self):
        self.rag_pipeline = None
        self._initialize()

    def _initialize(self):
        """Initialize the RAG pipeline."""
        try:
            # Initialize RAG pipeline
            self.rag_pipeline = RAGPipeline(
                embedding_model="all-MiniLM-L6-v2",
                llm_model="qwen3:8b",
                top_k_retrieval=3,
            )

            # Set up paths (relative to the competition directory)
            base_path = Path(__file__).parent.parent.parent  # Go up to project root
            topics_dir = base_path / "DM-i-AI-2025/emergency-healthcare-rag/data/topics"
            topics_json = (
                base_path / "DM-i-AI-2025/emergency-healthcare-rag/data/topics.json"
            )
            index_path = current_dir / "medical_index"

            # Setup the pipeline
            self.rag_pipeline.setup(str(topics_dir), str(topics_json), str(index_path))

            print("RAG classifier initialized successfully")

        except Exception as e:
            print(f"Error initializing RAG pipeline: {e}")
            print("Falling back to simple baseline")
            self.rag_pipeline = None

    def predict(self, statement: str) -> Tuple[int, int]:
        """
        Predict both binary classification and topic for a medical statement.

        Args:
            statement (str): The medical statement to classify

        Returns:
            Tuple[int, int]: (statement_is_true, statement_topic)
        """
        if self.rag_pipeline is not None:
            try:
                return self.rag_pipeline.predict(statement)
            except Exception as e:
                print(f"Error in RAG prediction: {e}")
                print("Falling back to baseline")

        # Fallback to simple baseline
        return self._baseline_predict(statement)

    def _baseline_predict(self, statement: str) -> Tuple[int, int]:
        """Simple baseline prediction."""
        import json

        # Always predict true for binary classification
        statement_is_true = 1

        # Simple keyword matching for topic
        try:
            base_path = Path(__file__).parent.parent.parent
            topics_json = (
                base_path / "DM-i-AI-2025/emergency-healthcare-rag/data/topics.json"
            )

            with open(topics_json, "r") as f:
                topics = json.load(f)

            statement_lower = statement.lower()
            best_topic = 0
            max_matches = 0

            for topic_name, topic_id in topics.items():
                keywords = (
                    topic_name.lower()
                    .replace("_", " ")
                    .replace("(", "")
                    .replace(")", "")
                    .split()
                )

                matches = sum(1 for keyword in keywords if keyword in statement_lower)

                if matches > max_matches:
                    max_matches = matches
                    best_topic = topic_id

            return statement_is_true, best_topic

        except Exception as e:
            print(f"Error in baseline prediction: {e}")
            return 1, 0


# Global classifier instance
_classifier = None


def get_classifier():
    """Get or create the global classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = MedicalStatementClassifier()
    return _classifier


def predict(statement: str) -> Tuple[int, int]:
    """
    Main prediction function that matches the original API.
    This function will be called by the competition framework.
    """
    classifier = get_classifier()
    return classifier.predict(statement)
