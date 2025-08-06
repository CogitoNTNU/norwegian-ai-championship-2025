#!/usr/bin/env python3
"""Profile where time is spent in the pipeline"""

import os
import sys
import time
from pathlib import Path

# Add rag-pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag-pipeline"))

from rag_pipeline_embeddings import EmbeddingsRAGPipeline

def profile_pipeline_step_by_step():
    """Profile each step of the pipeline"""

    print("Profiling RAG Pipeline Steps...")
    print("=" * 50)

    # Initialize
    pipeline = EmbeddingsRAGPipeline(
        embedding_model="pubmedbert-base-embeddings",
        llm_model="cogito:14b",
        retrieval_strategy="hyde",
        device="cuda"
    )

    rag_dir = Path(__file__).parent
    pipeline.setup(
        str(rag_dir / "data" / "topics"),
        str(rag_dir / "data" / "topics.json")
    )

    test_statement = "Aspirin reduces inflammation and prevents blood clots in patients with cardiovascular disease."

    print(f"\nTesting statement: {test_statement[:50]}...")

    # Step 1: Retrieval
    print("\n1. RETRIEVAL STEP:")
    start = time.time()
    relevant_chunks = pipeline.retrieval_strategy.retrieve(
        test_statement, pipeline.document_store, k=5
    )
    retrieval_time = time.time() - start
    print(f"   Time: {retrieval_time:.3f}s")
    print(f"   Retrieved {len(relevant_chunks)} chunks")

    # Step 2: Context building
    print("\n2. CONTEXT BUILDING:")
    start = time.time()
    context = pipeline._build_context(relevant_chunks)
    context_time = time.time() - start
    print(f"   Time: {context_time:.3f}s")
    print(f"   Context length: {len(context)} chars")

    # Step 3: LLM Classification
    print("\n3. LLM CLASSIFICATION:")
    likely_topics = pipeline._get_likely_topics(relevant_chunks)
    start = time.time()
    statement_is_true, statement_topic = pipeline.llm_client.classify_statement(
        test_statement, context, likely_topics
    )
    llm_time = time.time() - start
    print(f"   Time: {llm_time:.3f}s")
    print(f"   Result: True/False={statement_is_true}, Topic={statement_topic}")

    # Full pipeline for comparison
    print("\n4. FULL PIPELINE:")
    start = time.time()
    full_result = pipeline.predict(test_statement)
    full_time = time.time() - start
    print(f"   Time: {full_time:.3f}s")
    print(f"   Result: {full_result}")

    # Summary
    total_steps = retrieval_time + context_time + llm_time
    print("\n" + "=" * 50)
    print("TIME BREAKDOWN:")
    print("=" * 50)
    print(f"Retrieval:    {retrieval_time:.3f}s ({retrieval_time/total_steps*100:.1f}%)")
    print(f"Context:      {context_time:.3f}s ({context_time/total_steps*100:.1f}%)")
    print(f"LLM:          {llm_time:.3f}s ({llm_time/total_steps*100:.1f}%)")
    print(f"Total steps:  {total_steps:.3f}s")
    print(f"Full pipeline:{full_time:.3f}s")

    print(f"\nBottleneck: {'LLM' if llm_time > retrieval_time else 'Retrieval'}")

if __name__ == "__main__":
    profile_pipeline_step_by_step()
