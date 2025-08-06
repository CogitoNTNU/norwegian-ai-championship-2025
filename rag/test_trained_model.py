#!/usr/bin/env python3
"""
Test the trained medical embedding model.
"""

import os
from sentence_transformers import SentenceTransformer
import torch

def test_trained_model():
    """Test the trained medical embedding model."""
    model_path = "models/biobert-medical-embeddings_20250806_115150"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return False
    
    print("Testing Trained Medical Embedding Model")
    print("=" * 50)
    
    try:
        # Load the trained model
        print(f"Loading model from: {model_path}")
        model = SentenceTransformer(model_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Test medical queries
        medical_texts = [
            "Patient presents with chest pain and shortness of breath",
            "Acute myocardial infarction with ST elevation",
            "The patient has a history of hypertension and diabetes",
            "Administered aspirin and nitroglycerin for cardiac symptoms",
            "Echocardiogram shows reduced ejection fraction"
        ]
        
        print(f"\nGenerating embeddings for {len(medical_texts)} medical texts...")
        
        # Test different Matryoshka dimensions
        dimensions = [768, 512, 256, 128, 64]
        
        for dim in dimensions:
            print(f"\n📊 Testing dimension: {dim}")
            
            # Load model with specific dimension
            model_dim = SentenceTransformer(model_path, truncate_dim=dim)
            
            # Generate embeddings
            embeddings = model_dim.encode(medical_texts)
            
            print(f"   Embedding shape: {embeddings.shape}")
            print(f"   Sample embedding mean: {embeddings[0].mean():.4f}")
            print(f"   Sample embedding std: {embeddings[0].std():.4f}")
            
            # Test similarity between related medical texts
            similarity = model_dim.similarity(
                "chest pain and myocardial infarction",
                "heart attack with cardiac symptoms"
            )
            print(f"   Medical similarity score: {similarity[0][0]:.4f}")
        
        print("\n✅ Model testing completed successfully!")
        print("\nUsage examples:")
        print("```python")
        print("from sentence_transformers import SentenceTransformer")
        print(f"model = SentenceTransformer('{model_path}')")
        print("embeddings = model.encode(['your medical text here'])")
        print("")
        print("# For different dimensions:")
        print(f"model_256 = SentenceTransformer('{model_path}', truncate_dim=256)")
        print("```")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_trained_model()
    exit(0 if success else 1)