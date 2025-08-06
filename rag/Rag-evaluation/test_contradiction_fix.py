#!/usr/bin/env python3
"""
Test script to verify that the LLM can now properly detect contradictions
after our prompt engineering and deduplication fixes.
"""

import json
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from llm_client import LocalLLMClient
from templates.healthcare_rag import HealthcareRAG


def test_contradiction_detection():
    """Test the specific contradiction case that was failing."""
    
    print("🧪 Testing Contradiction Detection Fix")
    print("=" * 50)
    
    # Create LLM client
    llm_client = LocalLLMClient(model_name="cogito:3b")
    llm_client.ensure_model_available()
    
    # Test case from the original problem
    statement = "The radial approach in coronary angiography is associated with a higher risk of complications compared to the femoral approach."
    
    # Context that contradicts the statement
    context = """Access for angiography is gained via a large or medium-sized artery, and location varies according to the procedure. The femoral route is usually used as a retrograde approach for procedures involving the iliac vessels, the abdominal and thoracic aorta, the upper limbs, and the head and neck. Due to its large caliber, it allows for larger devices such as stents or occlusive aortic balloons. The radial approach is now commonly used in coronary angiography, as it comes with a lower risk of complications compared to the previous femoral or brachial routes. Larger size sheaths are easily accommodated via the femoral route, and often percutaneous suture devices and collagen plugs are used."""
    
    print(f"Statement: {statement}")
    print(f"\nContext: {context[:200]}...")
    print(f"\nExpected: FALSE (0) - because context says radial has 'lower risk', not 'higher risk'")
    
    # Test the LLM classification
    statement_is_true, statement_topic = llm_client.classify_statement(statement, context)
    
    print(f"\nLLM Response:")
    print(f"  Statement is true: {statement_is_true}")
    print(f"  Statement topic: {statement_topic}")
    
    # Check if the fix worked
    if statement_is_true == 0:
        print("\n✅ SUCCESS: LLM correctly identified the contradiction!")
        return True
    else:
        print("\n❌ FAILED: LLM still missed the contradiction.")
        return False


def test_healthcare_rag_deduplication():
    """Test that the healthcare RAG system properly deduplicates context."""
    
    print("\n🧪 Testing Healthcare RAG Deduplication")
    print("=" * 50)
    
    # Create healthcare RAG instance
    rag = HealthcareRAG()
    
    # Test with a query that might return duplicates
    query = "radial approach complications angiography"
    contexts = rag.retrieve_context(query, k=5)
    
    print(f"Query: {query}")
    print(f"Retrieved {len(contexts)} contexts")
    
    # Check for duplicates
    unique_contexts = set(contexts)
    if len(contexts) == len(unique_contexts):
        print("✅ SUCCESS: No duplicate contexts found!")
        return True
    else:
        print(f"❌ FAILED: Found {len(contexts) - len(unique_contexts)} duplicate contexts")
        return False


def test_full_pipeline():
    """Test the full pipeline with the problematic case."""
    
    print("\n🧪 Testing Full RAG Pipeline")
    print("=" * 50)
    
    # Create healthcare RAG instance
    rag = HealthcareRAG()
    
    # Test the problematic statement
    statement = "The radial approach in coronary angiography is associated with a higher risk of complications compared to the femoral approach."
    
    # Run the full pipeline
    result = rag.run(statement)
    
    # Parse the answer
    try:
        answer = json.loads(result["answer"])
        statement_is_true = answer.get("statement_is_true", 1)
        statement_topic = answer.get("statement_topic", 0)
        
        print(f"Statement: {statement}")
        print(f"Answer: {result['answer']}")
        print(f"Retrieved {len(result['context'])} contexts")
        
        # Check if the fix worked
        if statement_is_true == 0:
            print("\n✅ SUCCESS: Full pipeline correctly identified the contradiction!")
            return True
        else:
            print("\n❌ FAILED: Full pipeline still missed the contradiction.")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: Failed to parse answer: {e}")
        return False


def main():
    """Run all tests."""
    
    print("Testing Contradiction Detection Fixes")
    print("=" * 60)
    
    tests = [
        ("Direct LLM Test", test_contradiction_detection),
        ("Deduplication Test", test_healthcare_rag_deduplication),
        ("Full Pipeline Test", test_full_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All fixes are working correctly!")
    else:
        print("⚠️  Some issues remain. Check the output above.")


if __name__ == "__main__":
    main()
