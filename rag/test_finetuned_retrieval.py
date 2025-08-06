#!/usr/bin/env python3
"""
Quick test to verify fine-tuned model retrieval quality
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Rag-evaluation"))

from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

def test_retrieval_quality():
    print("Testing fine-tuned model retrieval...")
    
    # Load fine-tuned model
    model_path = os.path.join("models", "biobert-medical-final_20250806_161455")
    model = SentenceTransformer(model_path)
    print(f"Loaded model: {model_path}")
    
    # Load FAISS indexes for comparison
    old_index = faiss.read_index("Rag-evaluation/optimized_indexes/biobert_faiss.index")
    new_index = faiss.read_index("Rag-evaluation/optimized_indexes/biobert_medical_finetuned_faiss.index")
    
    # Load document mapping
    with open("Rag-evaluation/optimized_indexes/document_mapping.json", "r") as f:
        doc_mapping = json.load(f)
    
    # Test queries
    test_queries = [
        "What are the symptoms of acute myocardial infarction?",
        "How is sepsis diagnosed in the emergency department?", 
        "What is the treatment for pneumothorax?",
    ]
    
    print(f"\nTesting with {len(test_queries)} medical queries...")
    print("="*70)
    
    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: {query}")
        
        # Get embeddings
        query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
        
        # Search both indexes
        old_scores, old_indices = old_index.search(query_embedding, k=5)
        new_scores, new_indices = new_index.search(query_embedding, k=5)
        
        print("\nORIGINAL BioBERT (Top 3):")
        for j in range(min(3, len(old_indices[0]))):
            idx = old_indices[0][j]
            if idx < len(doc_mapping):
                doc_text = doc_mapping[idx]["text"][:200] + "..."
                print(f"  {j+1}. Score: {old_scores[0][j]:.4f} | {doc_text}")
        
        print("\nFINE-TUNED BioBERT (Top 3):")
        for j in range(min(3, len(new_indices[0]))):
            idx = new_indices[0][j]
            if idx < len(doc_mapping):
                doc_text = doc_mapping[idx]["text"][:200] + "..."
                print(f"  {j+1}. Score: {new_scores[0][j]:.4f} | {doc_text}")
        
        print("-" * 70)

if __name__ == "__main__":
    test_retrieval_quality()