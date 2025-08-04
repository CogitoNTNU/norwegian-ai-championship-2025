import ollama
import json
import os
import re
from typing import Dict, Tuple


class LocalLLMClient:
    def __init__(self, model_name: str = "mistral:7b-instruct"):
        self.model_name = model_name
        self.client = ollama.Client()

        # Load topic mapping
        self.topic_mapping = self._load_topic_mapping()

    def _load_topic_mapping(self) -> Dict[str, int]:
        """Load the topic mapping from the competition data."""
        try:
            # Try multiple possible paths
            possible_paths = [
                "data/topics.json",  # Direct path when running from rag root
                "../data/topics.json",
                "../../data/topics.json",
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "data", "topics.json"
                ),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load topic mapping: {e}. Using empty mapping.")
            return {}

        print(
            "Warning: Topic mapping file not found in any expected location. Using empty mapping."
        )
        return {}

    def ensure_model_available(self) -> None:
        """Ensure the model is downloaded and available."""
        try:
            # Try to list models and see if ours is there
            models = self.client.list()
            if hasattr(models, "models"):
                model_names = [model.model for model in models.models]

                if self.model_name not in model_names:
                    print(f"Model {self.model_name} not found. Pulling...")
                    self.client.pull(self.model_name)
                    print(f"Model {self.model_name} pulled successfully")
                else:
                    print(f"Model {self.model_name} is available")
            else:
                print(f"Could not list models, attempting to pull {self.model_name}...")
                self.client.pull(self.model_name)
                print(f"Model {self.model_name} pulled successfully")

        except Exception as e:
            print(f"Error checking/pulling model: {e}")
            print("Note: Make sure Ollama is running with 'ollama serve'")

    def classify_statement(self, statement: str, context: str) -> Tuple[int, int]:
        """
        Classify a medical statement using the LLM.

        Args:
            statement: The medical statement to classify
            context: Relevant medical context from RAG retrieval

        Returns:
            Tuple of (statement_is_true, statement_topic)
        """
        prompt = self._build_classification_prompt(statement, context)

        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.0,  # Deterministic outputs
                    "top_p": 0.1,  # Very focused sampling
                    "top_k": 1,  # Most likely token only
                    "num_predict": 50,  # Reduced from 100 for speed
                    "repeat_penalty": 1.0,  # No repeat penalty
                    "presence_penalty": 0.0,  # No presence penalty
                    "frequency_penalty": 0.0,  # No frequency penalty
                    "mirostat": 0,  # Disable mirostat sampling
                    "num_gpu": 1,  # Use GPU acceleration if available
                    "num_thread": 4,  # Reduced threads for faster single inference
                },
            )

            result_text = response["response"]
            return self._parse_classification_result(result_text)

        except Exception as e:
            print(f"Error in LLM classification: {e}")
            # Fallback: return neutral predictions
            return 1, 0

    def _build_classification_prompt(self, statement: str, context: str) -> str:
        """Build a classification prompt where LLM predicts topic names."""

        # Limit context to first 400 characters for faster processing
        context_limited = context[:400] if context else ""

        # Get topic names for the LLM to choose from
        topic_names = list(self.topic_mapping.keys())
        topics_str = ", ".join(topic_names)  # Include ALL topics

        # Create a more structured prompt with clearer instructions
        prompt = f"""Emergency Medical Classification Task:

Statement: "{statement}"

Relevant medical context:
{context_limited}

Instructions:
1. Determine if the statement is TRUE (1) or FALSE (0)
2. Identify the most relevant emergency medical topic from these options:
{topics_str}

Provide your answer in this exact JSON format:
{{"statement_is_true": 1, "statement_topic": "Pulmonary Embolism"}}

Replace the values with your classification."""

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

                # Handle topic - could be name or number
                topic_value = parsed.get("statement_topic", "Abdominal Trauma")

                if isinstance(topic_value, str):
                    # LLM returned topic name, convert to number
                    statement_topic = self._topic_name_to_number(topic_value)
                else:
                    # LLM returned number directly
                    statement_topic = int(topic_value)

                # Validate ranges
                statement_is_true = max(0, min(1, statement_is_true))
                statement_topic = max(0, min(114, statement_topic))

                return statement_is_true, statement_topic

            # Fallback parsing - try to extract topic name from text
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
                    # Try to find topic name in the line
                    for topic_name in self.topic_mapping.keys():
                        if topic_name.lower() in line.lower():
                            statement_topic = self.topic_mapping[topic_name]
                            break
                    else:
                        # Fallback to number extraction
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

    def _topic_name_to_number(self, topic_name: str) -> int:
        """Convert topic name to number using exact match or fuzzy matching."""
        # Exact match first
        if topic_name in self.topic_mapping:
            return self.topic_mapping[topic_name]

        # Fuzzy matching - try partial matches
        topic_name_lower = topic_name.lower()
        for name, number in self.topic_mapping.items():
            if topic_name_lower in name.lower() or name.lower() in topic_name_lower:
                return number

        # If no match found, return 0 (Abdominal Trauma as default)
        print(f"Warning: Could not match topic '{topic_name}', defaulting to 0")
        return 0
