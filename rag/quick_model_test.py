#!/usr/bin/env python3
"""
Quick test to verify custom embedding models are working correctly.
"""

import os
import sys
import time
from sentence_transformers import SentenceTransformer

def test_model(model_path: str):
    """Test a specific embedding model."""
    print(f"Testing model: {model_path}")
    print("-" * 60)
    
    # Load model
    print("Loading model...")
    start_time = time.time()
    try:
        model = SentenceTransformer(model_path)
        load_time = time.time() - start_time
        print(f"Model loaded successfully in {load_time:.2f}s")
        print(f"  Embedding dimensions: {model.get_sentence_embedding_dimension()}")
        print(f"  Device: {model.device}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Test embedding generation
    test_texts = [
        "Acute myocardial infarction is a medical emergency",
        "Sepsis kan diagnostiseres ved hjelp av qSOFA-score",
        "Pneumothorax treatment requires immediate intervention"
    ]
    
    print(f"\nTesting embedding generation with {len(test_texts)} samples...")
    start_time = time.time()
    try:
        embeddings = model.encode(test_texts, convert_to_numpy=True)
        encode_time = time.time() - start_time
        print(f"Embeddings generated successfully in {encode_time:.3f}s")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Sample values: {embeddings[0][:5]}...")
    except Exception as e:
        print(f"Failed to generate embeddings: {e}")
        return
    
    # Test similarity calculation
    print("\nTesting similarity calculations...")
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        print(f"Similarity calculation successful")
        print(f"  Similarity matrix shape: {similarities.shape}")
        print(f"  Sample similarities:")
        for i, text in enumerate(test_texts):
            print(f"    Text {i+1}: {text[:50]}...")
        print(f"  Similarity scores:")
        for i in range(len(test_texts)):
            for j in range(i+1, len(test_texts)):
                print(f"    Text {i+1} <-> Text {j+1}: {similarities[i][j]:.4f}")
    except Exception as e:
        print(f"Failed to calculate similarities: {e}")
    
    print(f"\nModel test completed successfully!")

def main():
    print("CUSTOM EMBEDDING MODEL QUICK TEST")
    print("=" * 60)
    
    # Test the models you have
    models_to_test = [
        "models/pubmedbert-medical-final_20250806_165702",
        "models/biobert-medical-final_20250806_161455",
    ]
    
    for model_path in models_to_test:
        full_path = os.path.join(os.path.dirname(__file__), model_path)
        if os.path.exists(full_path):
            print(f"\n{'='*60}")
            test_model(full_path)
        else:
            print(f"\n{'='*60}")
            print(f"Model not found: {full_path}")
    
    print(f"\n{'='*60}")
    print("All tests completed!")

if __name__ == "__main__":
    main()