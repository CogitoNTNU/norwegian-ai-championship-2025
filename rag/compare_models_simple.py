#!/usr/bin/env python3
"""
Simple comparison between original and fine-tuned BioBERT models
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Rag-evaluation"))

from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
from pathlib import Path

def compare_models():
    print("COMPARING ORIGINAL vs FINE-TUNED BioBERT")
    print("=" * 60)
    
    # Load models
    print("Loading models...")
    original_model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    finetuned_model = SentenceTransformer("models/biobert-medical-final_20250806_161455")
    
    # Load indexes
    print("Loading FAISS indexes...")
    original_index = faiss.read_index("Rag-evaluation/optimized_indexes/biobert_faiss.index")
    finetuned_index = faiss.read_index("Rag-evaluation/optimized_indexes/biobert_medical_finetuned_faiss.index")
    
    # Load document mapping
    with open("Rag-evaluation/optimized_indexes/document_mapping.json", "r") as f:
        doc_mapping = json.load(f)
    
    # Test with actual evaluation questions
    test_cases = [
        {
            "question": "Akutt hjerteinfarkt kan føre til død hvis ikke behandlet raskt",
            "expected_answer": True,
            "description": "Norwegian: Acute MI can lead to death if not treated quickly"
        },
        {
            "question": "Penicillin is the first-line treatment for all types of pneumonia",
            "expected_answer": False, 
            "description": "English: Penicillin first-line for all pneumonia types"
        },
        {
            "question": "Sepsis kan diagnostiseres ved hjelp av qSOFA-score",
            "expected_answer": True,
            "description": "Norwegian: Sepsis can be diagnosed using qSOFA score"
        },
        {
            "question": "All patients with chest pain should receive morphine immediately",
            "expected_answer": False,
            "description": "English: All chest pain patients get morphine"
        }
    ]
    
    print(f"\nTesting with {len(test_cases)} medical questions...")
    print("=" * 60)
    
    total_original_score = 0
    total_finetuned_score = 0
    
    for i, case in enumerate(test_cases):
        question = case["question"]
        expected = case["expected_answer"]
        desc = case["description"]
        
        print(f"\n{i+1}. {desc}")
        print(f"Question: {question}")
        print(f"Expected: {'TRUE' if expected else 'FALSE'}")
        
        # Get embeddings from both models
        orig_embedding = original_model.encode([question], convert_to_numpy=True).astype('float32')
        fine_embedding = finetuned_model.encode([question], convert_to_numpy=True).astype('float32')
        
        # Search both indexes
        orig_scores, orig_indices = original_index.search(orig_embedding, k=3)
        fine_scores, fine_indices = finetuned_index.search(fine_embedding, k=3)
        
        print(f"\nORIGINAL BioBERT - Top retrieved content:")
        for j in range(min(2, len(orig_indices[0]))):
            idx = orig_indices[0][j]
            if idx < len(doc_mapping):
                doc_text = doc_mapping[idx]["text"][:150] + "..."
                print(f"  Score: {orig_scores[0][j]:.3f} | {doc_text}")
        
        print(f"\nFINE-TUNED BioBERT - Top retrieved content:")
        for j in range(min(2, len(fine_indices[0]))):
            idx = fine_indices[0][j]
            if idx < len(doc_mapping):
                doc_text = doc_mapping[idx]["text"][:150] + "..."
                print(f"  Score: {fine_scores[0][j]:.3f} | {doc_text}")
        
        # Calculate average similarity scores (lower = better)
        orig_avg_score = np.mean(orig_scores[0][:3])
        fine_avg_score = np.mean(fine_scores[0][:3])
        
        total_original_score += orig_avg_score
        total_finetuned_score += fine_avg_score
        
        print(f"\nAverage Similarity Scores (lower = better):")
        print(f"Original: {orig_avg_score:.3f}")
        print(f"Fine-tuned: {fine_avg_score:.3f}")
        print(f"Improvement: {((orig_avg_score - fine_avg_score) / orig_avg_score * 100):.1f}%")
        print("-" * 60)
    
    # Overall comparison
    print(f"\nOVERALL RESULTS:")
    print(f"Original BioBERT avg score: {total_original_score/len(test_cases):.3f}")
    print(f"Fine-tuned BioBERT avg score: {total_finetuned_score/len(test_cases):.3f}")
    improvement = ((total_original_score - total_finetuned_score) / total_original_score * 100)
    print(f"Overall improvement: {improvement:.1f}%")
    
    if improvement > 0:
        print("✓ Fine-tuned model IS performing better!")
    else:
        print("✗ Fine-tuned model is NOT performing better")

if __name__ == "__main__":
    compare_models()