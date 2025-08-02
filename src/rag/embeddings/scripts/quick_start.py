#!/usr/bin/env python3
"""Quick start script to test the embeddings system with your RAG pipeline."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rag.embeddings.models import get_embedding_model
from rag.embeddings.utils import update_document_store


def main():
    """Quick start guide for using the embeddings system."""

    print("=" * 60)
    print("EMBEDDINGS SYSTEM QUICK START")
    print("=" * 60)

    print("\n1. Testing basic embedding functionality...")

    # Load a model
    model = get_embedding_model("nomic-embed-text-v1.5")

    # Test encoding
    test_statements = [
        "Testicular torsion requires immediate surgical intervention",
        "Patient presents with acute chest pain and dyspnea",
    ]

    embeddings = model.encode(test_statements)
    print(f"✓ Successfully encoded {len(test_statements)} statements")
    print(f"  Model: {model.get_model_name()}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")

    # Test Matryoshka dimensions
    if model.supports_matryoshka():
        print("\n2. Testing Matryoshka dimensions...")
        for dim in [512, 256]:
            emb = model.encode_with_dimension(test_statements[0], dimension=dim)
            print(f"  ✓ Dimension {dim}: shape {emb.shape}")

    print("\n3. Generating updated DocumentStore...")
    update_document_store()

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("\n1. To use the new embedding system in your RAG pipeline:")
    print("   cd src/rag/rag-pipeline")
    print("   cp document_store.py document_store_original.py  # Backup")
    print("   cp document_store_updated.py document_store.py    # Use new version")

    print("\n2. Or modify your classifier.py to specify a model:")
    print("   self.rag_pipeline = RAGPipeline(")
    print('       embedding_model="nomic-embed-text-v1.5",  # or any other model')
    print("       llm_model='qwen3:8b',")
    print("       top_k_retrieval=3")
    print("   )")

    print("\n3. To download more models:")
    print("   python src/rag/embeddings/scripts/download_models.py --all")

    print("\n4. To test different models:")
    print("   python src/rag/embeddings/scripts/test_embeddings.py")

    print("\n5. Available models:")
    from rag.embeddings.models import get_registry

    registry = get_registry()
    for model_name in registry.list_models()[:5]:
        print(f"   - {model_name}")
    print("   ... and more!")

    print("\n" + "=" * 60)
    print("Ready to use the new embeddings system!")
    print("=" * 60)


if __name__ == "__main__":
    main()
