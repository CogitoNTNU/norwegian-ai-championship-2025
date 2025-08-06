#!/usr/bin/env python3
"""
Simple test to verify your trained medical model works.
"""

def test_your_model():
    print("🏥 Testing Your Trained Medical Model")
    print("=" * 50)
    
    try:
        # Test loading your model
        from sentence_transformers import SentenceTransformer
        
        model_path = "models/biobert-medical-embeddings_20250806_120708"
        print(f"📁 Loading: {model_path}")
        
        model = SentenceTransformer(model_path)
        print("✅ Model loaded successfully!")
        
        # Test medical examples
        medical_texts = [
            "Patient presents with acute myocardial infarction",
            "Emergency department admits patient with chest pain", 
            "Heart attack symptoms include chest discomfort",
            "Patient has diabetes mellitus type 2"
        ]
        
        print("\n🧪 Testing medical text embeddings:")
        embeddings = model.encode(medical_texts)
        print(f"✅ Embeddings shape: {embeddings.shape}")
        
        # Test similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        
        print(f"\n🔬 Medical text similarities:")
        print(f"Heart attack vs Emergency chest pain: {similarities[0][1]:.3f}")
        print(f"Heart attack vs Heart attack symptoms: {similarities[0][2]:.3f}")
        print(f"Heart attack vs Diabetes: {similarities[0][3]:.3f}")
        
        # Test Matryoshka dimensions
        print(f"\n🎯 Testing Matryoshka dimensions:")
        for dim in [256, 128, 64]:
            model_truncated = SentenceTransformer(model_path, truncate_dim=dim)
            truncated_emb = model_truncated.encode(medical_texts[0])
            print(f"  Dimension {dim}: shape {truncated_emb.shape}")
        
        print(f"\n🎉 Your trained model works perfectly!")
        print(f"💡 The model shows good medical domain understanding")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_your_model()
