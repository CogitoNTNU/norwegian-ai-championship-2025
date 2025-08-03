#!/usr/bin/env python3
"""Test the embedding system with medical statements."""

import sys
from pathlib import Path
import time
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from rag.embeddings.models import get_embedding_model
from rag.embeddings.managers import ModelManager


def test_basic_functionality():
    """Test basic embedding functionality."""
    print("Testing basic embedding functionality...\n")

    # Test sentences
    test_sentences = [
        "Testicular torsion is a surgical emergency requiring immediate intervention.",
        "The patient presented with acute chest pain and shortness of breath.",
        "ECG showed ST-segment elevation in leads II, III, and aVF.",
    ]

    # Initialize model
    model = get_embedding_model("all-MiniLM-L6-v2")

    # Test encoding
    print("Testing encoding...")
    embeddings = model.encode(test_sentences, show_progress_bar=True)

    print(f"✓ Encoded {len(test_sentences)} sentences")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Dimension: {model.get_dimension()}")
    print(f"  Max sequence length: {model.get_max_seq_length()}")

    # Test similarity
    print("\nTesting similarity...")
    similarities = np.dot(embeddings, embeddings.T)
    print("Similarity matrix:")
    for i, sent1 in enumerate(test_sentences):
        for j, sent2 in enumerate(test_sentences):
            if i < j:
                print(
                    f"  '{sent1[:30]}...' vs '{sent2[:30]}...': {similarities[i, j]:.3f}"
                )


def test_matryoshka_model():
    """Test Matryoshka model with different dimensions."""
    print("\n\nTesting Matryoshka model...\n")

    model = get_embedding_model("nomic-embed-text-v1.5")

    test_text = (
        "Acute myocardial infarction requires immediate cardiac catheterization."
    )

    # Test different dimensions
    dimensions = [768, 512, 256, 128]

    print(f"Testing dimensions: {dimensions}")
    print(f"Test text: '{test_text}'")

    embeddings = {}

    for dim in dimensions:
        start_time = time.time()
        embedding = model.encode_with_dimension([test_text], dimension=dim)
        encode_time = time.time() - start_time

        embeddings[dim] = embedding[0]
        print(f"\n  Dimension {dim}:")
        print(f"    Encoding time: {encode_time:.3f}s")
        print(f"    Embedding norm: {np.linalg.norm(embedding[0]):.3f}")

    # Compare embeddings
    print("\nDimension reduction impact (cosine similarity with full embedding):")
    full_embedding = embeddings[768]
    for dim in [512, 256, 128]:
        truncated = embeddings[dim]
        # Pad truncated embedding for comparison
        padded = np.pad(truncated, (0, 768 - dim), mode="constant")
        similarity = np.dot(full_embedding, padded) / (
            np.linalg.norm(full_embedding) * np.linalg.norm(padded)
        )
        print(f"  {dim}d vs 768d: {similarity:.3f}")


def test_model_switching():
    """Test switching between models."""
    print("\n\nTesting model switching...\n")

    manager = ModelManager()

    models_to_test = ["all-MiniLM-L6-v2", "gte-base", "bge-base-en-v1.5"]
    test_text = (
        "Patient presents with acute appendicitis requiring surgical intervention."
    )

    embeddings = {}

    for model_name in models_to_test:
        print(f"Loading {model_name}...")
        model = manager.load_model(model_name)

        # Encode
        start_time = time.time()
        embedding = model.encode([test_text])
        encode_time = time.time() - start_time

        embeddings[model_name] = embedding[0]

        print(f"  ✓ Loaded and encoded in {encode_time:.3f}s")
        print(f"    Dimension: {model.get_dimension()}")
        print(f"    Embedding norm: {np.linalg.norm(embedding[0]):.3f}")

    # Compare embeddings across models
    print("\nCross-model similarities:")
    model_names = list(embeddings.keys())
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i < j:
                emb1 = embeddings[model1]
                emb2 = embeddings[model2]

                # Handle different dimensions
                min_dim = min(len(emb1), len(emb2))
                similarity = np.dot(emb1[:min_dim], emb2[:min_dim]) / (
                    np.linalg.norm(emb1[:min_dim]) * np.linalg.norm(emb2[:min_dim])
                )

                print(f"  {model1} vs {model2}: {similarity:.3f}")


def test_medical_preprocessing():
    """Test medical text preprocessing."""
    print("\n\nTesting medical text preprocessing...\n")

    # Test with and without preprocessing
    from rag.embeddings.models.medical_embeddings import medical_text_preprocessor

    test_texts = [
        "Patient with MI admitted to ICU for monitoring.",
        "ECG shows AFib with RVR, BP 140/90 mmHg.",
        "CT scan revealed PE, started on anticoagulation.",
    ]

    print("Original texts:")
    for text in test_texts:
        print(f"  - {text}")

    print("\nPreprocessed texts:")
    for text in test_texts:
        processed = medical_text_preprocessor(text)
        print(f"  - {processed}")

    # Compare embeddings
    model = get_embedding_model("all-MiniLM-L6-v2")

    print("\nEmbedding similarity (original vs preprocessed):")
    for text in test_texts:
        processed = medical_text_preprocessor(text)

        emb_original = model.encode([text])[0]
        emb_processed = model.encode([processed])[0]

        similarity = np.dot(emb_original, emb_processed) / (
            np.linalg.norm(emb_original) * np.linalg.norm(emb_processed)
        )

        print(f"  '{text[:30]}...': {similarity:.3f}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("EMBEDDING SYSTEM TEST SUITE")
    print("=" * 60)

    try:
        test_basic_functionality()
        test_matryoshka_model()
        test_model_switching()
        test_medical_preprocessing()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
