"""
Example of how to load and use your trained medical embedding model.
"""

from sentence_transformers import SentenceTransformer
import torch

def load_medical_model():
    """Load the trained medical embedding model."""
    
    # Path to your trained model
    model_path = "biobert-medical-embeddings-mrl"
    
    try:
        # Load the fine-tuned model
        model = SentenceTransformer(model_path)
        print(f"✅ Successfully loaded trained model from: {model_path}")
        print(f"📱 Model device: {model.device}")
        
        return model
    
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Falling back to base BioBERT model...")
        
        # Fallback to base model if trained model not found
        base_model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
        return base_model

def test_medical_embeddings():
    """Test the medical embedding model."""
    
    model = load_medical_model()
    
    # Test medical texts
    medical_texts = [
        "Patient presents with chest pain and shortness of breath",
        "Acute myocardial infarction with ST elevation in leads II, III, aVF",
        "Normal cardiac function on echocardiogram",
        "Pneumonia with consolidation in right lower lobe",
        "Patient has diabetes mellitus type 2"
    ]
    
    print("\n🧪 Testing medical text encoding...")
    
    # Encode texts with different dimensions (Matryoshka)
    dimensions = [768, 512, 256, 128, 64]
    
    for dim in dimensions:
        print(f"\n📏 Embedding dimension: {dim}")
        
        # Set truncation dimension for Matryoshka
        model.truncate_dim = dim
        
        # Generate embeddings
        embeddings = model.encode(medical_texts)
        print(f"   Shape: {embeddings.shape}")
        
        # Compute similarities
        similarities = model.similarity(embeddings, embeddings)
        
        # Show similarity between first two texts (both cardiac)
        cardiac_similarity = similarities[0][1].item()
        print(f"   Cardiac texts similarity: {cardiac_similarity:.4f}")
        
        # Show similarity between cardiac and non-cardiac
        cross_similarity = similarities[0][3].item()  # chest pain vs pneumonia
        print(f"   Cross-domain similarity: {cross_similarity:.4f}")

def integrate_with_rag():
    """Example of integrating with your RAG pipeline."""
    
    model = load_medical_model()
    
    # Example: Update your document store embeddings
    print("\n🔄 RAG Integration Example:")
    print("To use in your RAG pipeline, update document_store_embeddings.py:")
    print("""
    # In document_store_embeddings.py
    from sentence_transformers import SentenceTransformer
    
    class MedicalDocumentStore:
        def __init__(self):
            # Use your fine-tuned model instead of base model
            self.model = SentenceTransformer("biobert-medical-embeddings-mrl", truncate_dim=256)
            # ... rest of your implementation
    """)

if __name__ == "__main__":
    print("🏥 Medical Embedding Model Loader")
    print("=" * 50)
    
    # Test the model
    test_medical_embeddings()
    
    # Show integration example
    integrate_with_rag()
    
    print("\n✨ Ready to use your fine-tuned medical embeddings!")