#!/usr/bin/env python3
"""
Test the custom trained medical model with the RAG pipeline.
"""

import os
import sys

# Add rag-pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag-pipeline"))

def test_custom_model():
    """Test loading our custom medical model through the pipeline."""
    try:
        print("🏥 Testing Custom Medical Model with RAG Pipeline")
        print("=" * 60)
        
        from embeddings.models.sentence_transformers import SentenceTransformerModel
        
        # Test loading our custom model directly
        model_path = "models/biobert-medical-embeddings_20250806_120708"
        print(f"📁 Loading custom model: {model_path}")
        
        # Create model instance
        custom_model = SentenceTransformerModel(
            model_name=model_path,
            device="cuda" if os.environ.get("CUDA_AVAILABLE") != "false" else "cpu"
        )
        
        print(f"✅ Custom model loaded successfully!")
        print(f"📊 Model device: {custom_model.device}")
        
        # Test encoding
        test_text = "Patient presents with acute myocardial infarction"
        embedding = custom_model.encode([test_text])
        print(f"✅ Encoding test successful: {embedding.shape}")
        
        # Test with different dimensions
        if hasattr(custom_model.model, 'truncate_dim'):
            for dim in [256, 128]:
                try:
                    custom_model.model.truncate_dim = dim
                    truncated_embedding = custom_model.encode([test_text])
                    print(f"✅ Matryoshka {dim}D: {truncated_embedding.shape}")
                except Exception as e:
                    print(f"⚠️  Matryoshka {dim}D not available: {e}")
        
        print("\n🎉 Custom model integration successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_custom_model()
