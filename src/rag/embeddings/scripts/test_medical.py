#!/usr/bin/env python3
"""Test medical preprocessing and embeddings."""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from rag.embeddings.models import get_embedding_model
from rag.embeddings.preprocessing import preprocess_medical_text


def test_medical_preprocessing():
    """Test medical text preprocessing."""
    print("=" * 60)
    print("TESTING MEDICAL PREPROCESSING")
    print("=" * 60)

    test_cases = [
        "Pt presents w/ CP radiating to L arm, SOB noted",
        "ECG: ST↑ II, III, aVF → inferior MI",
        "GCS: E3V4M5 = 12",
        "ABG: pH 7.1, pCO2 25, HCO3- 10 → metabolic acidosis",
        "Rx: ASA 325mg PO STAT, NTG 0.4mg SL q5min x3",
    ]

    print("\nOriginal → Preprocessed:")
    print("-" * 60)

    for text in test_cases:
        preprocessed = preprocess_medical_text(text)
        print(f"{text}")
        print(f"→ {preprocessed}")
        print()


def test_medical_model():
    """Test medical embedding model."""
    print("\n" + "=" * 60)
    print("TESTING MEDICAL EMBEDDING MODEL")
    print("=" * 60)

    try:
        # Load medical model
        print("\nLoading pubmedbert-base-embeddings...")
        get_embedding_model("pubmedbert-base-embeddings")
        print("✓ Model loaded successfully")

        # Test medical statements
        medical_statements = [
            "Patient presents with acute myocardial infarction",
            "Administer epinephrine 0.3mg intramuscularly for anaphylaxis",
            "CT scan reveals subdural hematoma requiring urgent evacuation",
        ]

        # Test with and without preprocessing
        print("\nTesting embeddings with preprocessing:")

        # Create a medical model with preprocessing
        from rag.embeddings.models.sentence_transformers import MedicalEmbeddingModel

        medical_model = MedicalEmbeddingModel(
            "NeuML/pubmedbert-base-embeddings", preprocessing_fn=preprocess_medical_text
        )

        for statement in medical_statements:
            embedding = medical_model.encode(statement)
            print(f"✓ Encoded: '{statement[:50]}...' → shape {embedding.shape}")

    except Exception as e:
        print(f"✗ Error loading medical model: {e}")
        print("Note: This model may need to be downloaded first")


def compare_general_vs_medical():
    """Compare general vs medical models on medical text."""
    print("\n" + "=" * 60)
    print("COMPARING GENERAL VS MEDICAL MODELS")
    print("=" * 60)

    medical_texts = [
        "Acute ST-elevation myocardial infarction with cardiogenic shock",
        "Diabetic ketoacidosis with severe metabolic acidosis pH 7.1",
        "Tension pneumothorax requiring immediate needle decompression",
    ]

    general_texts = [
        "The patient is feeling unwell today",
        "The weather is sunny and warm",
        "The hospital is located downtown",
    ]

    try:
        # Load models
        print("\nLoading models...")
        general_model = get_embedding_model("all-MiniLM-L6-v2")
        print("✓ General model loaded")

        # Encode texts
        print("\nComputing embeddings...")
        medical_embeddings_general = general_model.encode(medical_texts)
        general_embeddings_general = general_model.encode(general_texts)

        # Compute average similarity within and between groups
        import numpy as np

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # Within medical texts
        medical_sims = []
        for i in range(len(medical_texts)):
            for j in range(i + 1, len(medical_texts)):
                sim = cosine_similarity(
                    medical_embeddings_general[i], medical_embeddings_general[j]
                )
                medical_sims.append(sim)

        # Between medical and general
        cross_sims = []
        for i in range(len(medical_texts)):
            for j in range(len(general_texts)):
                sim = cosine_similarity(
                    medical_embeddings_general[i], general_embeddings_general[j]
                )
                cross_sims.append(sim)

        print("\nResults with general model (all-MiniLM-L6-v2):")
        print(f"  Average similarity within medical texts: {np.mean(medical_sims):.3f}")
        print(
            f"  Average similarity between medical and general: {np.mean(cross_sims):.3f}"
        )
        print(f"  Difference: {np.mean(medical_sims) - np.mean(cross_sims):.3f}")

    except Exception as e:
        print(f"✗ Error in comparison: {e}")


def main():
    """Run all medical tests."""
    test_medical_preprocessing()
    test_medical_model()
    compare_general_vs_medical()

    print("\n" + "=" * 60)
    print("Medical testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
