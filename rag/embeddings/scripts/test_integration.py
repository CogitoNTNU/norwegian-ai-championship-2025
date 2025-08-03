#!/usr/bin/env python3
"""
Test the embeddings integration with training data.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
script_dir = Path(__file__).parent
embeddings_dir = script_dir.parent
rag_dir = embeddings_dir.parent
sys.path.insert(0, str(rag_dir))

from rag_pipeline.rag_pipeline_embeddings import EmbeddingsRAGPipeline
from Rag_evaluation.src.templates.embeddings_rag import EmbeddingsRAG
from embeddings.models import list_available_models


def test_pipeline_direct():
    """Test the RAG pipeline directly."""
    print("🧪 Testing RAG Pipeline Direct Integration")
    print("=" * 50)
    
    # Find data paths
    root_dir = Path(__file__).parent.parent.parent.parent
    topics_dir = root_dir / "data" / "topics"
    topics_json = root_dir / "data" / "topics.json"
    
    # Initialize pipeline
    print("\n1️⃣ Initializing pipeline...")
    pipeline = EmbeddingsRAGPipeline(
        embedding_model="all-MiniLM-L6-v2",
        llm_model="qwen3:8b",
        top_k_retrieval=5
    )
    
    # Setup pipeline
    print("2️⃣ Setting up pipeline with medical documents...")
    pipeline.setup(str(topics_dir), str(topics_json))
    
    # Test with a sample statement
    test_statement = "Diabetes is a condition that affects blood sugar levels"
    print(f"\n3️⃣ Testing with statement: '{test_statement}'")
    
    try:
        is_true, topic = pipeline.predict(test_statement)
        print(f"✅ Result: is_true={is_true}, topic={topic}")
        
        # Get model info
        info = pipeline.get_model_info()
        print(f"\n📊 Model Info:")
        print(f"   - Embedding Model: {info['embedding_model']}")
        print(f"   - LLM Model: {info['llm_model']}")
        print(f"   - Embedding Dimension: {info['dimension']}")
        print(f"   - Index Size: {info['index_size']} vectors")
        print(f"   - Number of Chunks: {info['num_chunks']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_evaluation_template():
    """Test the evaluation template integration."""
    print("\n\n🧪 Testing Evaluation Template Integration")
    print("=" * 50)
    
    try:
        # Initialize template
        print("\n1️⃣ Initializing EmbeddingsRAG template...")
        rag = EmbeddingsRAG(embedding_model="all-MiniLM-L6-v2")
        
        # Test with a sample statement
        test_statement = "Heart attacks require immediate medical attention"
        print(f"\n2️⃣ Testing with statement: '{test_statement}'")
        
        result = rag.run(test_statement)
        
        # Parse answer
        answer = json.loads(result["answer"])
        print(f"✅ Result: {answer}")
        print(f"   - Statement is true: {answer['statement_is_true']}")
        print(f"   - Statement topic: {answer['statement_topic']}")
        
        # Show context
        if result["context"]:
            print(f"\n📄 Retrieved contexts ({len(result['context'])} chunks):")
            for i, ctx in enumerate(result["context"][:2]):  # Show first 2
                print(f"\n   Context {i+1}: {ctx[:150]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_training_data():
    """Test with actual training data."""
    print("\n\n🧪 Testing with Training Data")
    print("=" * 50)
    
    # Find training data
    root_dir = Path(__file__).parent.parent.parent.parent
    statements_dir = root_dir / "data" / "train" / "statements"
    answers_dir = root_dir / "data" / "train" / "answers"
    
    if not statements_dir.exists() or not answers_dir.exists():
        print("❌ Training data not found")
        return
    
    # Load first few examples
    statement_files = sorted(list(statements_dir.glob("statement_*.txt")))[:5]
    
    # Initialize pipeline
    pipeline = EmbeddingsRAGPipeline()
    topics_dir = root_dir / "data" / "topics"
    topics_json = root_dir / "data" / "topics.json"
    pipeline.setup(str(topics_dir), str(topics_json))
    
    print(f"\n📊 Testing {len(statement_files)} training examples...")
    
    correct = 0
    for i, stmt_file in enumerate(statement_files):
        # Load statement
        with open(stmt_file, "r") as f:
            statement = f.read().strip()
        
        # Load answer
        answer_file = answers_dir / f"{stmt_file.stem}.json"
        with open(answer_file, "r") as f:
            true_answer = json.load(f)
        
        # Make prediction
        try:
            pred_is_true, pred_topic = pipeline.predict(statement)
            
            # Check if correct
            is_correct = (
                pred_is_true == true_answer["statement_is_true"] and
                pred_topic == true_answer["statement_topic"]
            )
            
            if is_correct:
                correct += 1
            
            print(f"\n{i+1}. Statement: {statement[:80]}...")
            print(f"   True: is_true={true_answer['statement_is_true']}, topic={true_answer['statement_topic']}")
            print(f"   Pred: is_true={pred_is_true}, topic={pred_topic}")
            print(f"   {'✅ Correct' if is_correct else '❌ Wrong'}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    accuracy = correct / len(statement_files) * 100
    print(f"\n📈 Accuracy: {correct}/{len(statement_files)} ({accuracy:.1f}%)")


def main():
    """Run all integration tests."""
    print("🚀 Embeddings Integration Test Suite")
    print("=" * 60)
    
    # Show available models
    print("\n📦 Available Embedding Models:")
    models = list_available_models()
    for model in models[:5]:  # Show first 5
        print(f"   - {model}")
    print(f"   ... and {len(models) - 5} more")
    
    # Run tests
    test_pipeline_direct()
    test_evaluation_template()
    test_training_data()
    
    print("\n\n✅ Integration tests completed!")


if __name__ == "__main__":
    main()