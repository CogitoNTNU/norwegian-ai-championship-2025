#!/usr/bin/env python3
"""
Test script to verify Google Gemma integration with LocalLLMClient
"""

import sys
from llm_client import LocalLLMClient


def test_google_gemma():
    """Test the Google Gemma integration."""
    print("🧪 Testing Google Gemma-3-4B-IT integration...")

    try:
        # Initialize the client
        print("📊 Initializing LocalLLMClient with Google Gemma...")
        client = LocalLLMClient()

        # Check if model is available
        print("🔍 Checking model availability...")
        client.ensure_model_available()

        # Test basic invocation
        print("💬 Testing basic invoke method...")
        test_prompt = "What is cardiac tamponade?"
        response = client.invoke(test_prompt)
        print(f"Response: {response[:100]}...")

        # Test classification method
        print("🏥 Testing classification method...")
        statement = "Hyperkalemia causes peaked T-waves on ECG"
        context = "Hyperkalemia is a condition with elevated potassium levels in blood. ECG changes include peaked T-waves, prolonged PR interval, and widened QRS complex."

        is_true, topic = client.classify_statement(statement, context)
        print(f"Classification result: is_true={is_true}, topic={topic}")

        print("✅ All tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False


if __name__ == "__main__":
    success = test_google_gemma()
    sys.exit(0 if success else 1)
