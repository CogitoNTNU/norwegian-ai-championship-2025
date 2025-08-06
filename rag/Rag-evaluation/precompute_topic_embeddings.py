"""
Precompute topic embeddings for fast topic classification.
This creates topic_embeddings.npy for the BiobertFast class.
"""

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    # Load document mapping to extract topics
    base_dir = "optimized_indexes"
    document_mapping_path = os.path.join(base_dir, "document_mapping.json")
    
    if not os.path.exists(document_mapping_path):
        print(f"Error: {document_mapping_path} not found. Run create_optimized_indexes.py first.")
        return
    
    with open(document_mapping_path, "r") as f:
        documents = json.load(f)
    
    # Extract unique topic names
    topic_names = list(set(doc['topic_name'] for doc in documents))
    print(f"Found {len(topic_names)} topics to embed")
    
    # Load BioBERT
    print("Loading BioBERT...")
    model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    
    # Compute embeddings
    print("Computing topic embeddings...")
    topic_embeddings = model.encode(
        topic_names, 
        batch_size=32,
        show_progress_bar=True
    )
    
    # Save embeddings
    output_path = os.path.join(base_dir, "topic_embeddings.npy")
    np.save(output_path, topic_embeddings)
    
    print(f"✅ Topic embeddings saved to {output_path}")
    print(f"Shape: {topic_embeddings.shape}")

if __name__ == "__main__":
    main()
