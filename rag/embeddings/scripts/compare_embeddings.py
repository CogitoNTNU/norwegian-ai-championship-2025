#!/usr/bin/env python3
"""
Compare different embedding models on the healthcare RAG task.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tabulate import tabulate
from tqdm import tqdm

# Add parent directory to path
script_dir = Path(__file__).parent
embeddings_dir = script_dir.parent
rag_dir = embeddings_dir.parent
sys.path.insert(0, str(rag_dir))

from rag_pipeline.rag_pipeline_embeddings import EmbeddingsRAGPipeline
from embeddings.models import MODEL_CONFIGS


def load_test_samples(
    statements_dir: str, 
    answers_dir: str, 
    max_samples: int = 50
) -> List[Tuple[str, Dict[str, int]]]:
    """Load test samples from training data."""
    samples = []
    
    statement_files = sorted(Path(statements_dir).glob("statement_*.txt"))[:max_samples]
    
    for statement_file in statement_files:
        answer_file = Path(answers_dir) / f"{statement_file.stem}.json"
        
        if answer_file.exists():
            with open(statement_file, "r") as f:
                statement = f.read().strip()
            
            with open(answer_file, "r") as f:
                answer = json.load(f)
            
            samples.append((statement, answer))
    
    return samples


def evaluate_model(
    model_name: str,
    test_samples: List[Tuple[str, Dict[str, int]]],
    topics_dir: str,
    topics_json: str
) -> Dict[str, Any]:
    """Evaluate a single embedding model."""
    print(f"\n🔍 Evaluating {model_name}...")
    
    # Initialize pipeline
    pipeline = EmbeddingsRAGPipeline(
        embedding_model=model_name,
        llm_model="qwen3:8b",
        top_k_retrieval=5
    )
    
    # Setup timing
    start_setup = time.time()
    pipeline.setup(topics_dir, topics_json)
    setup_time = time.time() - start_setup
    
    # Evaluate on samples
    correct_binary = 0
    correct_topic = 0
    correct_both = 0
    total_time = 0
    
    for statement, true_answer in tqdm(test_samples, desc=f"Testing {model_name}"):
        start_pred = time.time()
        
        try:
            pred_binary, pred_topic = pipeline.predict(statement)
            
            if pred_binary == true_answer["statement_is_true"]:
                correct_binary += 1
            
            if pred_topic == true_answer["statement_topic"]:
                correct_topic += 1
            
            if (pred_binary == true_answer["statement_is_true"] and 
                pred_topic == true_answer["statement_topic"]):
                correct_both += 1
        
        except Exception as e:
            print(f"Error processing statement: {e}")
        
        total_time += time.time() - start_pred
    
    # Calculate metrics
    n = len(test_samples)
    return {
        "model": model_name,
        "setup_time": setup_time,
        "avg_inference_time": total_time / n,
        "total_time": setup_time + total_time,
        "binary_accuracy": correct_binary / n,
        "topic_accuracy": correct_topic / n,
        "both_accuracy": correct_both / n,
        "model_info": pipeline.get_model_info()
    }


def main():
    """Main comparison function."""
    print("🏥 Healthcare RAG Embedding Model Comparison")
    print("=" * 60)
    
    # Find data paths
    script_path = Path(__file__)
    root_dir = script_path.parent.parent.parent.parent
    
    topics_dir = root_dir / "data" / "topics"
    topics_json = root_dir / "data" / "topics.json"
    statements_dir = root_dir / "data" / "train" / "statements"
    answers_dir = root_dir / "data" / "train" / "answers"
    
    # Verify paths exist
    if not all(p.exists() for p in [topics_dir, topics_json, statements_dir, answers_dir]):
        print("❌ Error: Could not find required data directories")
        print(f"  Topics dir: {topics_dir} (exists: {topics_dir.exists()})")
        print(f"  Topics JSON: {topics_json} (exists: {topics_json.exists()})")
        print(f"  Statements: {statements_dir} (exists: {statements_dir.exists()})")
        print(f"  Answers: {answers_dir} (exists: {answers_dir.exists()})")
        return
    
    # Load test samples
    print("\n📊 Loading test samples...")
    test_samples = load_test_samples(str(statements_dir), str(answers_dir), max_samples=30)
    print(f"Loaded {len(test_samples)} test samples")
    
    # Models to compare
    models_to_test = [
        "all-MiniLM-L6-v2",        # Fast baseline
        "all-mpnet-base-v2",       # Higher quality
        "multi-qa-MiniLM-L6-cos-v1",  # Q&A optimized
        "e5-small-v2",             # E5 family
        "bge-small-en-v1.5",       # BGE model
        "gte-small",               # GTE model
    ]
    
    # Filter to only available models
    available_models = [m for m in models_to_test if m in MODEL_CONFIGS]
    print(f"\n🤖 Testing {len(available_models)} models: {', '.join(available_models)}")
    
    # Evaluate each model
    results = []
    for model_name in available_models:
        try:
            result = evaluate_model(
                model_name, 
                test_samples,
                str(topics_dir),
                str(topics_json)
            )
            results.append(result)
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
    
    # Sort by combined accuracy
    results.sort(key=lambda x: x["both_accuracy"], reverse=True)
    
    # Display results
    print("\n📈 Results Summary")
    print("=" * 80)
    
    # Prepare table data
    table_data = []
    for r in results:
        table_data.append([
            r["model"],
            f"{r['model_info']['dimension']}",
            f"{r['setup_time']:.1f}s",
            f"{r['avg_inference_time']*1000:.1f}ms",
            f"{r['binary_accuracy']*100:.1f}%",
            f"{r['topic_accuracy']*100:.1f}%",
            f"{r['both_accuracy']*100:.1f}%"
        ])
    
    headers = ["Model", "Dim", "Setup", "Avg Time", "Binary Acc", "Topic Acc", "Both Acc"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Best model recommendation
    best_model = results[0]
    print(f"\n🏆 Best Model: {best_model['model']}")
    print(f"   - Combined Accuracy: {best_model['both_accuracy']*100:.1f}%")
    print(f"   - Binary Accuracy: {best_model['binary_accuracy']*100:.1f}%")
    print(f"   - Topic Accuracy: {best_model['topic_accuracy']*100:.1f}%")
    print(f"   - Average Inference Time: {best_model['avg_inference_time']*1000:.1f}ms")
    
    # Save results
    output_file = script_dir / "embedding_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Full results saved to: {output_file}")


if __name__ == "__main__":
    main()