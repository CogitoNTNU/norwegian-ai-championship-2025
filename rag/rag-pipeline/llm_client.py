import ollama
import json
import re
from typing import Dict, Tuple


class LocalLLMClient:
    def __init__(self, model_name: str = "qwen3:8b"):
        self.model_name = model_name
        self.client = ollama.Client()

        # Load topic mapping
        self.topic_mapping = self._load_topic_mapping()

    def _load_topic_mapping(self) -> Dict[str, int]:
        """Load the topic mapping from the competition data."""
        try:
            from pathlib import Path

            # Go up two levels from llm_client.py to the rag folder, then one more to the project root
            project_root = Path(__file__).parent.parent.parent
            topics_json_path = project_root / "data" / "topics.json"
            with open(topics_json_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print("Warning: Could not load topic mapping. Using empty mapping.")
            return {}

    def ensure_model_available(self) -> None:
        """Ensure the model is downloaded and available."""
        try:
            # Try to list models and see if ours is there
            models = self.client.list()["models"]
            model_names = [model["name"] for model in models]

            if self.model_name not in model_names:
                print(f"Model {self.model_name} not found. Pulling...")
                self.client.pull(self.model_name)
                print(f"Model {self.model_name} pulled successfully")
            else:
                print(f"Model {self.model_name} is available")

        except Exception as e:
            print(f"Error checking/pulling model: {e}")

    def classify_statement(
        self, statement: str, context: str, likely_topics: list = None
    ) -> Tuple[int, int]:
        """
        Classify a medical statement using the LLM.

        Args:
            statement: The medical statement to classify
            context: Relevant medical context from RAG retrieval
            likely_topics: A list of (topic_id, topic_name) tuples

        Returns:
            Tuple of (statement_is_true, statement_topic)
        """
        prompt = self._build_classification_prompt(statement, context, likely_topics)

        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,  # Low temperature for consistent outputs
                    "top_p": 0.9,
                    "num_predict": 100,  # Limit output length for speed
                },
            )

            result_text = response["response"]
            return self._parse_classification_result(result_text)

        except Exception as e:
            print(f"Error in LLM classification: {e}")
            # Fallback: return neutral predictions
            return 1, 0

    def _build_classification_prompt(
        self, statement: str, context: str, likely_topics: list
    ) -> str:
        """Build the classification prompt."""

        # Create topic list for reference
        if likely_topics:
            topic_list = "\n".join([f"{tid}: {tname}" for tid, tname in likely_topics])
            topic_guidance = (
                f"Focus on the following topics (and their IDs):\n{topic_list}"
            )
        else:
            topic_list = ""
            for name, idx in sorted(self.topic_mapping.items(), key=lambda x: x[1]):
                topic_list += f"{idx}: {name}\n"
            topic_guidance = f"Choose from the full list of topics:\n{topic_list}"

        prompt = f"""You are a medical expert analyzing emergency healthcare statements. 

CONTEXT (relevant medical information):
{context}

MEDICAL STATEMENT TO ANALYZE:
"{statement}"

TASK: Determine if the statement is TRUE or FALSE, and identify the most relevant medical topic.

AVAILABLE TOPICS:
{topic_guidance}

INSTRUCTIONS:
1. Based on the context and your medical knowledge, determine if the statement is factually TRUE (1) or FALSE (0)
2. Identify which topic ID the statement is most closely related to.
3. Respond ONLY in this exact JSON format:

{{"statement_is_true": 0 or 1, "statement_topic": topic_id}}

Your response:"""

        return prompt

    def _parse_classification_result(self, result_text: str) -> Tuple[int, int]:
        """Parse the LLM response to extract classification results."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r"\{.*?\}", result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)

                statement_is_true = int(parsed.get("statement_is_true", 1))
                statement_topic = int(parsed.get("statement_topic", 0))

                # Validate ranges
                statement_is_true = max(0, min(1, statement_is_true))
                statement_topic = max(0, min(114, statement_topic))

                return statement_is_true, statement_topic

            # Fallback parsing
            lines = result_text.strip().split("\n")
            statement_is_true = 1
            statement_topic = 0

            for line in lines:
                if "true" in line.lower() and ("false" in line.lower() or "0" in line):
                    statement_is_true = (
                        0
                        if ("false" in line.lower() or '"statement_is_true": 0' in line)
                        else 1
                    )
                if "topic" in line.lower():
                    topic_match = re.search(r"(\d+)", line)
                    if topic_match:
                        statement_topic = int(topic_match.group(1))
                        statement_topic = max(0, min(114, statement_topic))

            return statement_is_true, statement_topic

        except Exception as e:
            print(f"Error parsing LLM result: {e}")
            print(f"Raw result: {result_text}")
            # Return safe defaults
            return 1, 0
