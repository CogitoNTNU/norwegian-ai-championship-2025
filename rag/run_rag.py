#!/usr/bin/env python3
"""
Simple CLI script to run the RAG pipeline with a medical statement.
"""

import os

# Disable multiprocessing for sentence transformers on macOS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import argparse
from pathlib import Path

# Add rag-pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag-pipeline"))

from rag_pipeline_embeddings import EmbeddingsRAGPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG pipeline on a medical statement"
    )
    parser.add_argument(
        "--statement",
        type=str,
        required=True,
        help="Medical statement to classify"
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="pubmedbert-base-embeddings",
        help="Embedding model to use (default: pubmedbert-base-embeddings)"
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="cogito:8b",
        help="LLM model to use (default: cogito:8b)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of relevant chunks to retrieve (default: 5)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="default",
        choices=["default", "hyde"],
        help="Retrieval strategy to use (default: default)"
    )
    
    args = parser.parse_args()
    
    print(f"🏥 Medical Statement RAG Pipeline")
    print(f"=" * 50)
    print(f"📝 Statement: '{args.statement}'")
    print(f"🧬 Embedding Model: {args.embedding}")
    print(f"🤖 LLM Model: {args.llm}")
    print(f"🔍 Top-K Retrieval: {args.top_k}")
    print(f"🔍 Strategy: {args.strategy}")
    print()
    
    # Initialize pipeline
    print("⚙️  Initializing pipeline...")
    pipeline = EmbeddingsRAGPipeline(
        embedding_model=args.embedding,
        llm_model=args.llm,
        top_k_retrieval=args.top_k,
        retrieval_strategy=args.strategy
    )
    
    # Setup with data
    rag_dir = Path(__file__).parent
    topics_dir = rag_dir / "data" / "topics"
    topics_json = rag_dir / "data" / "topics.json"
    
    pipeline.setup(str(topics_dir), str(topics_json))
    
    # Make prediction
    print("\n🔮 Making prediction...")
    try:
        statement_is_true, statement_topic = pipeline.predict(args.statement)
        
        print(f"\n✅ Results:")
        print(f"   Statement is true: {statement_is_true} ({'True' if statement_is_true else 'False'})")
        print(f"   Statement topic: {statement_topic}")
        
        # Try to get topic name if topics.json exists
        if topics_json.exists():
            import json
            with open(topics_json) as f:
                topics = json.load(f)
            topic_name = next((name for name, id in topics.items() if id == statement_topic), "Unknown")
            print(f"   Topic name: {topic_name}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())