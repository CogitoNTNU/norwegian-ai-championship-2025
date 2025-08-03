#!/usr/bin/env python3
"""Simple test to verify each model loads and encodes correctly."""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from rag.embeddings.models import get_embedding_model, get_registry


def test_model(model_name):
    """Test a single model."""
    print(f"\nTesting {model_name}...")
    try:
        # Load model
        model = get_embedding_model(model_name)
        print("  ✓ Model loaded")

        # Test encoding
        test_text = "Patient presents with acute chest pain"
        embedding = model.encode(test_text)
        print("  ✓ Encoding successful")
        print(f"    Dimension: {embedding.shape[-1]}")

        # Test Matryoshka if supported
        if model.supports_matryoshka():
            dims = model.get_matryoshka_dimensions()
            print(f"  ✓ Matryoshka dimensions: {dims}")

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Test all available models."""
    print("=" * 60)
    print("TESTING INDIVIDUAL MODELS")
    print("=" * 60)

    registry = get_registry()
    models = registry.list_models()

    # Test a subset of models
    test_models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "gte-base",
        "bge-base-en-v1.5",
        "pubmedbert-base-embeddings",
    ]

    successful = 0
    failed = 0

    for model_name in test_models:
        if model_name in models:
            if test_model(model_name):
                successful += 1
            else:
                failed += 1

    print("\n" + "=" * 60)
    print(f"SUMMARY: {successful} successful, {failed} failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
