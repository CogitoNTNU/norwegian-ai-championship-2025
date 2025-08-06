import requests
import json
import os
from typing import Dict, Tuple


class LocalLLMClient:
    def __init__(
        self, model_name: str = "cogito:14b", ollama_url: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url

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
        """Check if the Ollama server is available."""
        try:
            # Check if server is available by hitting the main endpoint
            response = requests.get(f"{self.ollama_url}/", timeout=5)
            if response.status_code == 200:
                print(f"Ollama server is available at {self.ollama_url}")
            else:
                print(f"Warning: Ollama server returned status {response.status_code}")
        except Exception as e:
            print(f"Error checking Ollama server: {e}")
            print(f"Note: Make sure Ollama server is running at {self.ollama_url}")

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
            # Use Ollama generate endpoint with JSON format
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "top_p": 0.1,
                        "top_k": 1,
                        "num_predict": 50,
                        "repeat_penalty": 1.0,
                    },
                },
                timeout=30,
            )

            if response.status_code != 200:
                print(
                    f"Error from Ollama server: {response.status_code} - {response.text}"
                )
                return 1, 0

            response_data = response.json()
            result_text = response_data.get("response", "")
            return self._parse_classification_result(result_text)

        except Exception as e:
            print(f"Error in LLM classification: {e}")
            # Fallback: return neutral predictions
            return 1, 0

    def _build_classification_prompt(self, statement: str, context: str) -> str:
        """Build a focused classification prompt for better accuracy."""

        # Format the topic mapping into a readable string for the prompt
        topics_text = "\n".join(
            f"- {name}: {num}" for name, num in self.topic_mapping.items()
        )

        prompt = f"""You are a medical fact-checker tasked with evaluating medical statements for accuracy and categorization.

Available Medical Topics (with numbers):
{topics_text}

Provided Medical Context:
{context}

Statement to Evaluate: {statement}

Instructions:
1. FACT-CHECKING: Use ONLY the provided medical context to verify if the statement is accurate.
   - If the context supports the statement as medically correct, mark as true (1)
   - If the context contradicts the statement or shows it's incorrect, mark as false (0)
   - Base your decision strictly on the evidence in the provided context

2. TOPIC CLASSIFICATION: Identify which medical topic (by number) the statement primarily relates to:
   - Look for key medical terms, conditions, procedures, or concepts in the statement
   - Match these to the most appropriate topic from the numbered list above
   - Choose the topic number that best represents the main medical focus

3. OUTPUT FORMAT: Respond with valid JSON containing exactly these fields:
   - "is_true": 1 (if statement is accurate per context) or 0 (if inaccurate per context)
   - "topic": the topic number (0-114) that best matches the statement's medical focus

Example Response:
{{
    "is_true": 1,
    "topic": 42
}}

Provide ONLY the JSON response, no additional text."""

        return prompt

    def _parse_classification_result(self, result_text: str) -> Tuple[int, int]:
        """Parse the LLM response to extract classification results."""
        try:
            # When format: 'json' is used, the response should be a valid JSON string.
            parsed = json.loads(result_text)

            statement_is_true = int(
                parsed.get("statement_is_true", parsed.get("is_true", 1))
            )

            # Handle topic - could be name or number
            topic_value = parsed.get("statement_topic", parsed.get("topic", 0))

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

        except (json.JSONDecodeError, AttributeError) as e:
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
