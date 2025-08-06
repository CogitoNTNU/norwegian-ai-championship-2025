#!/usr/bin/env python3
"""
Simple test to verify FAISS segfault fix works on ARM Mac.
Tests the OptimizedSmartRAG without running the full evaluation.
"""

import sys
import time
from pathlib import Path

# Add the templates directory to the path
sys.path.append(str(Path(__file__).parent / "templates"))

def test_faiss_safety():
    print("🧪 Testing FAISS safety fixes...")
    
    try:
        from optimized_smart_rag import OptimizedSmartRAG
        from llm_client import LocalLLMClient
        
        print("✅ Imports successful")
        
        # Initialize the system (this loads FAISS)
        print("🚀 Initializing OptimizedSmartRAG...")
        llm_client = LocalLLMClient()
        rag_system = OptimizedSmartRAG(llm_client)
        print("✅ Initialization successful")
        
        # Test a simple question
        test_question = "Acute appendicitis is most commonly caused by lymphoid hyperplasia in pediatric patients."
        
        print(f"\n❓ Test Question: {test_question}")
        print("⏱️  Processing with FAISS safety checks...")
        
        start_time = time.time()
        result = rag_system.run(test_question)
        end_time = time.time()
        
        print(f"\n✅ Success! No segfault occurred.")
        print(f"   Processing time: {end_time - start_time:.2f} seconds")
        print(f"   Answer: {result['answer']}")
        print(f"   Retrieved {len(result['context'])} context documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_faiss_minimal():
    """Test minimal FAISS operations to isolate issues."""
    print("\n🔬 Testing minimal FAISS operations...")
    
    try:
        import faiss
        import numpy as np
        
        # Configure for ARM Mac
        faiss.omp_set_num_threads(1)
        
        # Create minimal test data
        d = 128
        xb = np.random.random((100, d)).astype('float32')
        xq = np.random.random((5, d)).astype('float32')
        
        # Ensure C-contiguous
        xb = np.ascontiguousarray(xb)
        xq = np.ascontiguousarray(xq)
        
        print(f"   Test data: xb={xb.shape}, xq={xq.shape}")
        print(f"   Data types: xb={xb.dtype}, xq={xq.dtype}")
        print(f"   Contiguous: xb={xb.flags['C_CONTIGUOUS']}, xq={xq.flags['C_CONTIGUOUS']}")
        
        # Create index
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        
        # Search
        distances, indices = index.search(xq, 5)
        
        print("✅ Minimal FAISS test successful")
        print(f"   Found {len(indices[0])} results per query")
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal FAISS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 FAISS Segfault Fix Test")
    print("=" * 50)
    
    # Test minimal FAISS first
    minimal_success = test_faiss_minimal()
    
    if minimal_success:
        # Test full system
        full_success = test_faiss_safety()
        
        if full_success:
            print("\n🎉 All tests passed! FAISS segfault appears to be fixed.")
        else:
            print("\n⚠️ Minimal FAISS works, but full system failed.")
    else:
        print("\n💥 Minimal FAISS test failed - this suggests a deeper compatibility issue.")
        print("Consider reinstalling FAISS with: brew install faiss")
