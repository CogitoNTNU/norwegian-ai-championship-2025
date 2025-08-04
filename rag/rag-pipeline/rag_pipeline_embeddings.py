"""RAG Pipeline using configurable embedding models."""

from typing import Tuple
from document_store_embeddings import EmbeddingsDocumentStore
from llm_client import LocalLLMClient
from retrieval_strategies import DefaultRetrieval, HyDERetrieval, HybridRetrieval


class EmbeddingsRAGPipeline:
    """RAG Pipeline with configurable embedding models."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "qwen3:8b",
        top_k_retrieval: int = 5,
        retrieval_strategy: str = "default",
    ):
        """
        Initialize RAG pipeline with specified embedding model.

        Args:
            embedding_model: Model name from embeddings registry
            llm_model: Local LLM model name
            top_k_retrieval: Number of relevant chunks to retrieve
            retrieval_strategy: Strategy to use for retrieval ("default" or "hyde")
        """
        self.document_store = EmbeddingsDocumentStore(embedding_model)
        self.llm_client = LocalLLMClient(llm_model)
        self.top_k = top_k_retrieval
        self.embedding_model = embedding_model

        # Set up retrieval strategy
        if retrieval_strategy == "hyde":
            self.retrieval_strategy = HyDERetrieval(self.llm_client)
        elif retrieval_strategy == "hybrid":
            self.retrieval_strategy = HybridRetrieval(alpha=0.5)
        else:
            self.retrieval_strategy = DefaultRetrieval()

    def setup(
        self, topics_dir: str, topics_json: str, index_path: str | None = None
    ) -> None:
        """
        Set up the RAG pipeline by loading documents and building index.

        Args:
            topics_dir: Path to medical topics directory
            topics_json: Path to topics mapping JSON
            index_path: Path to save/load pre-built index (optional)
        """
        print(f"Setting up RAG pipeline with {self.embedding_model}...")

        # Ensure LLM model is available
        print("Checking LLM availability...")
        self.llm_client.ensure_model_available()
        print(f"✅ LLM {self.llm_client.model_name} is available")

        # Load documents (shared across models)
        print("Loading medical documents...")
        self.document_store.load_medical_documents(topics_dir, topics_json)

        # Build or load index (model-specific)
        print("Building/loading embeddings index...")
        self.document_store.build_index()

        print("RAG pipeline ready!")

    def predict(self, statement: str) -> Tuple[int, int]:
        """
        Make prediction for a medical statement.

        Args:
            statement: Medical statement to classify

        Returns:
            Tuple of (statement_is_true, statement_topic)
        """
        # Step 1: Retrieve relevant context using the strategy
        relevant_chunks = self.retrieval_strategy.retrieve(
            statement, self.document_store, k=self.top_k
        )

        # Step 2: Build context from retrieved chunks
        context = self._build_context(relevant_chunks)

        # Step 3: Use LLM to classify with context
        # Also pass the most likely topics based on retrieval
        likely_topics = self._get_likely_topics(relevant_chunks)
        statement_is_true, statement_topic = self.llm_client.classify_statement(
            statement, context, likely_topics
        )

        return statement_is_true, statement_topic

    def _build_context(self, relevant_chunks: list) -> str:
        """Build context string from retrieved chunks."""
        if not relevant_chunks:
            return "No relevant medical context found."

        context_pieces = []
        seen_topics = set()
        topic_counts = {}

        # Build context and track topics
        for chunk_data in relevant_chunks:
            chunk = chunk_data["chunk"]
            metadata = chunk_data["metadata"]
            topic_name = metadata["topic_name"]
            topic_id = metadata["topic_id"]
            score = chunk_data["score"]

            # Track topic occurrences for classification
            topic_counts[topic_id] = topic_counts.get(topic_id, 0) + score

            # Add topic diversity - include chunks from different topics
            if topic_name not in seen_topics or len(context_pieces) < 3:
                context_pieces.append(f"[{topic_name}] {chunk}")
                seen_topics.add(topic_name)

            # Limit total context length for speed
            if len("\n\n".join(context_pieces)) > 1500:
                break

        # Store most likely topic for potential use
        if topic_counts:
            self._most_likely_topic = max(topic_counts.items(), key=lambda x: x[1])[0]
        else:
            self._most_likely_topic = 0

        return "\n\n".join(context_pieces)

    def _get_likely_topics(self, relevant_chunks: list) -> list:
        """Get most likely topics based on retrieval scores."""
        topic_scores = {}

        for chunk_data in relevant_chunks:
            metadata = chunk_data["metadata"]
            topic_id = metadata["topic_id"]
            topic_name = metadata["topic_name"]
            score = chunk_data["score"]

            if topic_id not in topic_scores:
                topic_scores[topic_id] = {"name": topic_name, "score": 0}
            topic_scores[topic_id]["score"] += score

        # Sort by score and return top topics
        sorted_topics = sorted(
            topic_scores.items(), key=lambda x: x[1]["score"], reverse=True
        )[:5]  # Top 5 most likely topics

        return [(tid, tdata["name"]) for tid, tdata in sorted_topics]

    def evaluate_on_training_data(
        self,
        training_statements_dir: str,
        training_answers_dir: str,
        max_samples: int = 10,
    ) -> dict:
        """
        Evaluate pipeline on training data for debugging.

        Args:
            training_statements_dir: Directory with statement files
            training_answers_dir: Directory with answer files
            max_samples: Maximum number of samples to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        from pathlib import Path
        import json

        statements_path = Path(training_statements_dir)
        answers_path = Path(training_answers_dir)

        correct_binary = 0
        correct_topic = 0
        correct_both = 0
        total = 0

        statement_files = sorted(list(statements_path.glob("statement_*.txt")))[
            :max_samples
        ]

        print(f"Evaluating {self.embedding_model} on {len(statement_files)} samples...")

        for statement_file in statement_files:
            # Get corresponding answer file
            statement_id = statement_file.stem  # e.g., "statement_0001"
            answer_file = answers_path / f"{statement_id}.json"

            if not answer_file.exists():
                continue

            # Load statement and answer
            with open(statement_file, "r") as f:
                statement = f.read().strip()

            with open(answer_file, "r") as f:
                true_answer = json.load(f)

            # Make prediction
            try:
                pred_binary, pred_topic = self.predict(statement)

                # Check accuracy
                if pred_binary == true_answer["statement_is_true"]:
                    correct_binary += 1

                if pred_topic == true_answer["statement_topic"]:
                    correct_topic += 1

                if (
                    pred_binary == true_answer["statement_is_true"]
                    and pred_topic == true_answer["statement_topic"]
                ):
                    correct_both += 1

                total += 1

                print(
                    f"{statement_id}: Binary {pred_binary}={true_answer['statement_is_true']} "
                    f"Topic {pred_topic}={true_answer['statement_topic']}"
                )

            except Exception as e:
                print(f"Error processing {statement_file}: {e}")

        if total == 0:
            return {"error": "No samples processed"}

        return {
            "model": self.embedding_model,
            "total_samples": total,
            "binary_accuracy": correct_binary / total,
            "topic_accuracy": correct_topic / total,
            "both_accuracy": correct_both / total,
            "correct_binary": correct_binary,
            "correct_topic": correct_topic,
            "correct_both": correct_both,
        }

    def get_model_info(self) -> dict:
        """Get information about the current configuration."""
        return {
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_client.model_name,
            "top_k": self.top_k,
            **self.document_store.get_model_info(),
        }
