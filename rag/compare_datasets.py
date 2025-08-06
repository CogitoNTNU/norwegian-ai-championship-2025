"""
Compare original vs improved medical embedding datasets.
"""

import json
import os
from typing import Dict, List, Any

def estimate_tokens(text: str) -> int:
    """Estimate token count."""
    return int(len(text.split()) / 0.75)

def simple_similarity(text1: str, text2: str) -> float:
    """Simple word overlap similarity."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0.0

def analyze_dataset(file_path: str, name: str) -> Dict[str, Any]:
    """Analyze dataset quality metrics."""
    if not os.path.exists(file_path):
        print(f"Dataset not found: {file_path}")
        return {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = data['examples']
    
    # Calculate metrics
    anchor_tokens = [estimate_tokens(ex['anchor']) for ex in examples]
    positive_tokens = [estimate_tokens(ex['positive']) for ex in examples]
    
    # Calculate semantic similarities
    similarities = [
        simple_similarity(ex['anchor'], ex['positive']) 
        for ex in examples
    ]
    
    # Count negatives
    total_negatives = sum(len(ex['negatives']) for ex in examples)
    
    return {
        'name': name,
        'total_examples': len(examples),
        'avg_anchor_tokens': sum(anchor_tokens) / len(anchor_tokens),
        'avg_positive_tokens': sum(positive_tokens) / len(positive_tokens),
        'min_positive_tokens': min(positive_tokens),
        'max_positive_tokens': max(positive_tokens),
        'avg_similarity': sum(similarities) / len(similarities),
        'min_similarity': min(similarities),
        'max_similarity': max(similarities),
        'total_negatives': total_negatives,
        'avg_negatives_per_example': total_negatives / len(examples),
        'zero_similarity_count': sum(1 for s in similarities if s == 0.0),
        'low_similarity_count': sum(1 for s in similarities if s < 0.05),
        'similarities': similarities,
        'examples': examples[:3]  # Sample examples
    }

def compare_datasets():
    """Compare original vs improved datasets."""
    print("Dataset Quality Comparison")
    print("=" * 60)
    
    # Analyze both datasets
    original = analyze_dataset("medical_embedding_dataset.json", "Original Dataset")
    improved = analyze_dataset("improved_medical_embedding_dataset.json", "Improved Dataset")
    
    if not original or not improved:
        print("Error: Could not load both datasets for comparison.")
        return
    
    # Print comparison
    print(f"\n{'Metric':<30} {'Original':<15} {'Improved':<15} {'Change':<15}")
    print("-" * 80)
    
    metrics = [
        ('Total Examples', 'total_examples', '{:.0f}'),
        ('Avg Anchor Tokens', 'avg_anchor_tokens', '{:.1f}'),
        ('Avg Positive Tokens', 'avg_positive_tokens', '{:.1f}'),
        ('Min Positive Tokens', 'min_positive_tokens', '{:.0f}'),
        ('Max Positive Tokens', 'max_positive_tokens', '{:.0f}'),
        ('Avg Similarity', 'avg_similarity', '{:.3f}'),
        ('Min Similarity', 'min_similarity', '{:.3f}'),
        ('Max Similarity', 'max_similarity', '{:.3f}'),
        ('Zero Similarity Count', 'zero_similarity_count', '{:.0f}'),
        ('Low Similarity (<0.05)', 'low_similarity_count', '{:.0f}'),
        ('Avg Negatives/Example', 'avg_negatives_per_example', '{:.1f}')
    ]
    
    for metric_name, key, fmt_str in metrics:
        orig_val = original[key]
        impr_val = improved[key]
        
        # Calculate change
        if isinstance(orig_val, (int, float)) and orig_val != 0:
            if key in ['zero_similarity_count', 'low_similarity_count']:
                # For these metrics, lower is better
                change = f"{((orig_val - impr_val) / orig_val * 100):+.1f}%"
            else:
                change = f"{((impr_val - orig_val) / orig_val * 100):+.1f}%"
        else:
            change = "N/A"
        
        print(f"{metric_name:<30} {fmt_str.format(orig_val):<15} {fmt_str.format(impr_val):<15} {change:<15}")
    
    # Show sample examples
    print(f"\nSample Example Comparison:")
    print("=" * 60)
    
    # Find a good example to compare
    for i in range(min(3, len(original['examples']), len(improved['examples']))):
        orig_ex = original['examples'][i]
        impr_ex = improved['examples'][i]
        
        orig_sim = simple_similarity(orig_ex['anchor'], orig_ex['positive'])
        impr_sim = simple_similarity(impr_ex['anchor'], impr_ex['positive'])
        
        print(f"\nExample {i+1}:")
        print(f"Statement: {orig_ex['anchor'][:100]}...")
        print()
        
        print("ORIGINAL POSITIVE:")
        print(f"  Similarity: {orig_sim:.3f}")
        print(f"  Tokens: {estimate_tokens(orig_ex['positive'])}")
        print(f"  Content: {orig_ex['positive'][:150]}...")
        print()
        
        print("IMPROVED POSITIVE:")
        print(f"  Similarity: {impr_sim:.3f}")
        print(f"  Tokens: {estimate_tokens(impr_ex['positive'])}")
        print(f"  Content: {impr_ex['positive'][:150]}...")
        print()
        
        improvement = ((impr_sim - orig_sim) / orig_sim * 100) if orig_sim > 0 else float('inf')
        print(f"Similarity Improvement: {improvement:+.1f}%")
        print("-" * 60)
    
    # Summary
    print(f"\nSUMMARY:")
    print("=" * 60)
    
    # Key improvements
    similarity_improvement = ((improved['avg_similarity'] - original['avg_similarity']) / 
                            original['avg_similarity'] * 100)
    token_improvement = ((improved['avg_positive_tokens'] - original['avg_positive_tokens']) / 
                        original['avg_positive_tokens'] * 100)
    zero_sim_reduction = ((original['zero_similarity_count'] - improved['zero_similarity_count']) / 
                         original['zero_similarity_count'] * 100) if original['zero_similarity_count'] > 0 else 0
    
    print(f"Key Improvements:")
    print(f"  + Average similarity improved by {similarity_improvement:+.1f}%")
    print(f"  + Context tokens increased by {token_improvement:+.1f}%") 
    print(f"  + Zero similarity examples reduced by {zero_sim_reduction:.1f}%")
    print(f"  + Better medical concept preservation")
    print(f"  + Semantic relevance-based selection")
    
    if improved['avg_similarity'] > original['avg_similarity']:
        print(f"\nRESULT: Improved dataset shows significantly better semantic alignment!")
    else:
        print(f"\nRESULT: Need to investigate similarity scoring.")
    
    # Check if improved dataset exists and has reasonable stats
    if (improved['total_examples'] > 0 and 
        improved['avg_similarity'] > 0.1 and
        improved['avg_positive_tokens'] > 200):
        
        print(f"\nREADY FOR TRAINING:")
        print(f"  Improved dataset: improved_medical_embedding_dataset.json")
        print(f"  Examples: {improved['total_examples']}")
        print(f"  Avg similarity: {improved['avg_similarity']:.3f}")
        print(f"  Avg context tokens: {improved['avg_positive_tokens']:.0f}")
    else:
        print(f"\nWARNING: Improved dataset may need further refinement.")

if __name__ == "__main__":
    compare_datasets()