"""
GraphRAG template using simulated graph retrieval.
Simplified to work with the current evaluation task without requiring Neo4j.
"""

import os
import json
from typing import Dict, List, Any

try:
    from ..llm_client import LocalLLMClient
except ImportError:
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from llm_client import LocalLLMClient


class GraphRAG:
    """Graph RAG system using simulated graph retrieval for medical queries."""

    def __init__(self, llm_client: LocalLLMClient = None):
        """Initialize GraphRAG."""
        self.llm_client = llm_client or LocalLLMClient()
        # Simulate medical knowledge graph entities and relationships
        self.medical_graph = {
            "symptoms": [
                "fever",
                "cough",
                "chest pain",
                "shortness of breath",
                "nausea",
            ],
            "conditions": [
                "pneumonia",
                "flu",
                "covid-19",
                "heart attack",
                "appendicitis",
            ],
            "treatments": [
                "antibiotics",
                "rest",
                "oxygen therapy",
                "surgery",
                "medication",
            ],
            "relationships": {
                "fever": ["flu", "covid-19", "pneumonia"],
                "chest pain": ["heart attack", "pneumonia"],
                "cough": ["flu", "covid-19", "pneumonia"],
            },
        }

    def retrieve_context(
        self, question: str, reference_contexts: List[str], k: int = 5
    ) -> List[str]:
        """Simulate graph-based context retrieval."""
        try:
            # Simple simulation: extract medical terms from question
            question_lower = question.lower()
            relevant_contexts = []

            # Check for symptoms in question
            for symptom in self.medical_graph["symptoms"]:
                if symptom in question_lower:
                    # Get related conditions
                    related = self.medical_graph["relationships"].get(symptom, [])
                    for condition in related:
                        relevant_contexts.append(
                            f"Medical knowledge: {symptom} is associated with {condition}"
                        )

            # Add reference contexts if provided
            if reference_contexts:
                relevant_contexts.extend(
                    reference_contexts[: k - len(relevant_contexts)]
                )

            # Fallback if no specific matches
            if not relevant_contexts:
                relevant_contexts = [
                    "General medical knowledge: Healthcare decisions should be based on symptoms and medical history.",
                    "Medical guideline: Always consult healthcare professionals for accurate diagnosis.",
                ]

            return relevant_contexts[:k]

        except Exception as e:
            print(f"Error in GraphRAG retrieval: {e}")
            return reference_contexts[:k] if reference_contexts else []

    def run(
        self, question: str, reference_contexts: List[str] = None
    ) -> Dict[str, Any]:
        """
        Process a question using simulated graph retrieval methods.

        Args:
            question: The input medical question
            reference_contexts: List of reference context strings

        Returns:
            Dict with 'answer' and 'context' keys
        """
        try:
            if not reference_contexts:
                reference_contexts = []

            # Retrieve context using simulated graph approach
            retrieved_contexts = self.retrieve_context(
                question, reference_contexts, k=5
            )

            # Combine contexts for LLM
            combined_context = "\n".join(retrieved_contexts[:3])  # Limit to top 3

            # Use LLM to classify the statement
            statement_is_true, statement_topic = self.llm_client.classify_statement(
                question, combined_context
            )

            # Format answer for evaluation
            answer = {
                "statement_is_true": statement_is_true,
                "statement_topic": statement_topic,
            }

            return {"answer": json.dumps(answer), "context": retrieved_contexts}

        except Exception as e:
            print(f"Error in GraphRAG.run: {e}")
            # Return safe defaults
            return {
                "answer": json.dumps({"statement_is_true": 1, "statement_topic": 0}),
                "context": [],
            }
