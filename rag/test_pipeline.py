#!/usr/bin/env python3
"""
Test the RAG pipeline with multiple statements from training data.
"""

import os
import platform

# Configure multiprocessing based on platform
if platform.system() == "Darwin":  # macOS
    # Disable multiprocessing for sentence transformers on macOS
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"  # Using 1 for macOS to avoid issues
else:  # Windows/Linux
    # Enable multiprocessing for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # Let system decide optimal thread count
    if "OMP_NUM_THREADS" not in os.environ:
        import multiprocessing

        os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())

# Set custom cache directory for sentence-transformers
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(os.path.dirname(__file__), ".cache", "sentence_transformers")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add rag-pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag-pipeline"))

from rag_pipeline_embeddings import EmbeddingsRAGPipeline


def load_training_statements(n: int) -> List[Tuple[str, Dict]]:
    """Load n statements from training data."""
    data_dir = Path(__file__).parent / "data"
    statements_dir = data_dir / "train" / "statements"
    answers_dir = data_dir / "train" / "answers"

    samples = []
    files = sorted(statements_dir.glob("statement_*.txt"))[:n]

    for stmt_file in files:
        answer_file = answers_dir / f"{stmt_file.stem}.json"

        if answer_file.exists():
            with open(stmt_file, "r") as f:
                statement = f.read().strip()

            with open(answer_file, "r") as f:
                answer = json.load(f)

            samples.append((statement, answer))

    return samples


def main():
    parser = argparse.ArgumentParser(description="Test RAG pipeline with multiple statements")
    parser.add_argument("--n", type=int, default=15, help="Number of statements to test (default: 15)")
    parser.add_argument(
        "--embedding",
        type=str,
        default="pubmedbert-base-embeddings",
        help="Embedding model to use",
    )
    parser.add_argument("--llm", type=str, default="cogito:8b", help="LLM model to use")
    parser.add_argument(
        "--strategy",
        type=str,
        default="default",
        choices=["default", "hyde", "hybrid"],
        help="Retrieval strategy to use (default: default)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show detailed output for each statement")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use: auto (default), cpu, or cuda",
    )

    args = parser.parse_args()

    # Check if using external LLM
    ollama_host = os.environ.get("OLLAMA_HOST")

    # Check CUDA availability and device selection
    import torch

    cuda_available = torch.cuda.is_available()

    print("[DEBUG] Debug Info:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Platform: {platform.system()}")
    print(f"   Requested device: {args.device}")

    if args.device == "auto":
        if platform.system() == "Darwin":  # macOS
            selected_device = "cpu"
            device_reason = "CPU (macOS default to avoid MPS issues)"
        else:
            selected_device = "cuda" if cuda_available else "cpu"
            device_reason = f"{'CUDA' if cuda_available else 'CPU'} (auto-detected)"
    elif args.device == "cuda":
        if cuda_available:
            selected_device = "cuda"
            device_reason = "CUDA (forced)"
        else:
            print("[WARNING] CUDA requested but not available, falling back to CPU")
            selected_device = "cpu"
            device_reason = "CPU (CUDA not available)"
    else:
        selected_device = "cpu"
        device_reason = "CPU (forced)"

    print("[TESTING] Testing RAG Pipeline")
    print("=" * 50)
    print(f"📊 Statements: {args.n}")
    print(f"🧬 Embedding: {args.embedding}")

    if ollama_host:
        print(f"🤖 LLM: External server at {ollama_host}")
        print(f"   (--llm argument '{args.llm}' will be ignored)")
    else:
        print(f"🤖 LLM: {args.llm} (local)")

    print(f"🔍 Strategy: {args.strategy}")
    print(f"[DEVICE] Device: {device_reason}")
    if cuda_available:
        print(f"[CUDA] CUDA Info: {torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} GPU(s))")
    print()

    # Load statements
    print(f"📂 Loading {args.n} statements...")
    samples = load_training_statements(args.n)

    if not samples:
        print("❌ No statements found!")
        return 1

    # Initialize pipeline
    print("\n⚙️  Initializing pipeline...")
    start_init = time.time()

    pipeline = EmbeddingsRAGPipeline(
        embedding_model=args.embedding,
        llm_model=args.llm,
        top_k_retrieval=5,
        retrieval_strategy=args.strategy,
        device=selected_device,
    )

    # Check what device the embedding model is actually using
    actual_device = pipeline.document_store.embedding_model.device
    print(f"[DEVICE] Embedding model using: {actual_device}")

    # Setup
    rag_dir = Path(__file__).parent
    pipeline.setup(str(rag_dir / "data" / "topics"), str(rag_dir / "data" / "topics.json"))

    init_time = time.time() - start_init
    print(f"✅ Pipeline ready in {init_time:.1f}s")

    # Test statements
    print(f"\n🔍 Testing {len(samples)} statements...\n")

    correct_binary = 0
    correct_topic = 0
    correct_both = 0
    total_time = 0

    for i, (statement, true_answer) in enumerate(samples):
        start_pred = time.time()

        try:
            # Make prediction
            pred_binary, pred_topic = pipeline.predict(statement)
            pred_time = time.time() - start_pred
            total_time += pred_time

            # Check correctness
            is_binary_correct = pred_binary == true_answer["statement_is_true"]
            is_topic_correct = pred_topic == true_answer["statement_topic"]

            if is_binary_correct:
                correct_binary += 1
            if is_topic_correct:
                correct_topic += 1
            if is_binary_correct and is_topic_correct:
                correct_both += 1

            # Display results
            if args.verbose:
                print(f"Statement {i + 1}:")
                print(f"  '{statement[:80]}...'")
                print(
                    f"  Binary: {pred_binary} (true: {true_answer['statement_is_true']}) {'✓' if is_binary_correct else '✗'}"
                )
                print(
                    f"  Topic:  {pred_topic} (true: {true_answer['statement_topic']}) {'✓' if is_topic_correct else '✗'}"
                )
                print(f"  Time:   {pred_time:.2f}s")
                print()
            else:
                status = (
                    "✓✓"
                    if is_binary_correct and is_topic_correct
                    else "✓✗"
                    if is_binary_correct
                    else "✗✓"
                    if is_topic_correct
                    else "✗✗"
                )
                print(
                    f"[{i + 1:3d}/{len(samples)}] {status} Binary: {pred_binary}/{true_answer['statement_is_true']}, Topic: {pred_topic:3d}/{true_answer['statement_topic']:3d} ({pred_time:.1f}s)"
                )

        except Exception as e:
            print(f"[{i + 1:3d}/{len(samples)}] ❌ Error: {str(e)[:50]}...")

    # Summary
    n = len(samples)
    avg_time = total_time / n if n > 0 else 0
    total_score = (correct_binary + correct_topic) / (2 * n) if n > 0 else 0

    print(f"\n{'=' * 50}")
    print("📊 RESULTS SUMMARY")
    print(f"{'=' * 50}")
    print(f"Binary Accuracy:    {correct_binary}/{n} ({correct_binary / n * 100:.1f}%)")
    print(f"Topic Accuracy:     {correct_topic}/{n} ({correct_topic / n * 100:.1f}%)")
    print(f"Combined Accuracy:  ({correct_binary} + {correct_topic}) / {2 * n} = {total_score:.1%}")
    print(f"Average Time:       {avg_time:.2f}s per statement")
    print(f"Total Time:         {total_time:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
