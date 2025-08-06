#!/usr/bin/env python3
"""
Test script to verify the optimized SmartRAG works without building indexes dynamically.
"""

import sys
import time
from pathlib import Path

# Add the templates directory to the path
sys.path.append(str(Path(__file__).parent / "templates"))

from optimized_smart_rag import OptimizedSmartRAG
from llm_client import LocalLLMClient

def test_optimized_rag():
    print("🧪 Testing OptimizedSmartRAG...")
    
    # Initialize the system
    llm_client = LocalLLMClient()
    rag_system = OptimizedSmartRAG(llm_client)
    
    # Test question
    test_question = "Acute appendicitis is most commonly caused by lymphoid hyperplasia in pediatric patients."
    
    print(f"\n❓ Test Question: {test_question}")
    print("⏱️  Processing...")
    
    start_time = time.time()
    result = rag_system.run(test_question)
    end_time = time.time()
    
    print(f"\n✅ Result:")
    print(f"   Answer: {result['answer']}")
    print(f"   Processing time: {end_time - start_time:.2f} seconds")
    print(f"   Context sources: {len(result['context'])} documents")
    
    # Show first context document
    if result['context']:
        print(f"\n📄 First context snippet:")
        print(f"   {result['context'][0][:200]}...")

if __name__ == "__main__":
    test_optimized_rag()
