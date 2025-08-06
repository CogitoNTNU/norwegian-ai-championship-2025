#!/usr/bin/env python3
"""
Debug script to analyze why BioBERT semantic search is retrieving less relevant documents
for binary classification compared to BM25s.
"""

import sys
import os
import json
import random

# Add paths for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from templates.biobert_rag import BiobertRAG
from templates.smart_rag import SmartRAG
from llm_client import LocalLLMClient


def load_test_cases(data_dir: str, project_root: str, num_samples: int = 10):
    """Load a sample of test cases for analysis"""
    statements_dir = os.path.join(data_dir, "statements")
    answers_dir = os.path.join(data_dir, "answers")

    # Load topic mapping
    topics_path = os.path.join(project_root, "data", "topics.json")
    try:
        with open(topics_path, "r") as f:
            topics = json.load(f)
            id_to_topic = {v: k for k, v in topics.items()}
    except Exception:
        id_to_topic = {}

    test_cases = []
    for stmt_file in sorted(os.listdir(statements_dir))[:100]:  # First 100 files
        if stmt_file.endswith(".txt"):
            stmt_id = stmt_file.replace(".txt", "")
            answer_file = os.path.join(answers_dir, f"{stmt_id}.json")

            if os.path.exists(answer_file):
                # Read statement
                with open(
                    os.path.join(statements_dir, stmt_file), "r", encoding="utf-8"
                ) as f:
                    statement = f.read().strip()

                # Read answer
                with open(answer_file, "r", encoding="utf-8") as f:
                    answer = json.load(f)

                topic_id = answer.get("statement_topic", 0)
                topic_name = id_to_topic.get(topic_id, "Unknown")

                test_cases.append(
                    {
                        "question": statement,
                        "ground_truth": str(answer.get("statement_is_true", 1)),
                        "reference_contexts": [topic_name],
                        "statement_id": stmt_id,
                    }
                )

    # Return a random sample
    random.seed(42)
    return random.sample(test_cases, min(num_samples, len(test_cases)))


def analyze_specific_examples():
    """Analyze specific problematic examples"""

    # Known problematic statements for analysis
    test_cases = [
        {
            "question": "The radial approach for coronary angiography carries a higher risk of complications compared to the femoral or brachial routes.",
            "ground_truth": "0",  # False
            "reference_contexts": ["Angiography (invasive)"],
        },
        {
            "question": "ARDS mortality rates increase with disease severity: 45% for mild, 32% for moderate, and 27% for severe disease respectively.",
            "ground_truth": "0",  # False - the order is wrong
            "reference_contexts": ["Acute Respiratory Distress Syndrome"],
        },
        {
            "question": "MRI uses ionizing radiation to produce high-quality images with superior soft-tissue contrast based on the magnetic properties of hydrogen molecules.",
            "ground_truth": "0",  # False - MRI doesn't use ionizing radiation
            "reference_contexts": ["MRI"],
        },
    ]

    print("Initializing diagnostic system...")
    llm_client = LocalLLMClient(model_name="cogito:14b")
    llm_client.ensure_model_available()
    diagnostic = BiobertRAG(llm_client=llm_client)

    print("Starting analysis of specific examples...")

    for i, case in enumerate(test_cases):
        print(f"\n{'#' * 100}")
        print(f"ANALYZING CASE {i + 1}")
        print(f"{'#' * 100}")

        diagnostic.analyze_question(
            case["question"], case["ground_truth"], case["reference_contexts"]
        )

        # Let's also get the actual answers from each method
        print("\n📊 GETTING ACTUAL LLM RESPONSES:")

        # Get semantic-only results
        semantic_results = diagnostic.semantic_only_retrieval(case["question"], k=5)
        semantic_context = "\n".join(
            [doc["document"]["text"] for doc in semantic_results]
        )

        # Get BM25-only results
        bm25_results = diagnostic.bm25_only_retrieval(case["question"], k=5)
        bm25_context = "\n".join([doc["document"]["text"] for doc in bm25_results])

        # Get topic-classified BM25 results
        topic_bm25_results = diagnostic.topic_classified_bm25_retrieval(
            case["question"], k=5
        )
        topic_bm25_context = "\n".join(
            [doc["document"]["text"] for doc in topic_bm25_results]
        )

        # Get hybrid results
        hybrid_results = diagnostic.hybrid_retrieval(case["question"], k=5)
        hybrid_context = "\n".join([doc["document"]["text"] for doc in hybrid_results])

        print("\n🔍 SEMANTIC-ONLY LLM RESPONSE:")
        try:
            semantic_response = llm_client.classify_statement(
                case["question"], semantic_context
            )
            print(
                f"   Predicted: {semantic_response[0]} (Ground Truth: {case['ground_truth']})"
            )
            print(
                f"   Correct: {'✅' if str(semantic_response[0]) == case['ground_truth'] else '❌'}"
            )
        except Exception as e:
            print(f"   Error: {e}")

        print("\n🔑 BM25-ONLY LLM RESPONSE:")
        try:
            bm25_response = llm_client.classify_statement(
                case["question"], bm25_context
            )
            print(
                f"   Predicted: {bm25_response[0]} (Ground Truth: {case['ground_truth']})"
            )
            print(
                f"   Correct: {'✅' if str(bm25_response[0]) == case['ground_truth'] else '❌'}"
            )
        except Exception as e:
            print(f"   Error: {e}")

        print("\n🎯 TOPIC-CLASSIFIED BM25 LLM RESPONSE:")
        try:
            # Show which topic was classified
            if topic_bm25_results:
                classified_topic = topic_bm25_results[0].get(
                    "classified_topic", "Unknown"
                )
                topic_doc_count = topic_bm25_results[0].get("topic_doc_count", 0)
                print(
                    f"   Classified Topic: {classified_topic} ({topic_doc_count} docs)"
                )

            topic_bm25_response = llm_client.classify_statement(
                case["question"], topic_bm25_context
            )
            print(
                f"   Predicted: {topic_bm25_response[0]} (Ground Truth: {case['ground_truth']})"
            )
            print(
                f"   Correct: {'✅' if str(topic_bm25_response[0]) == case['ground_truth'] else '❌'}"
            )
        except Exception as e:
            print(f"   Error: {e}")

        print("\n🔄 HYBRID LLM RESPONSE:")
        try:
            hybrid_response = llm_client.classify_statement(
                case["question"], hybrid_context
            )
            print(
                f"   Predicted: {hybrid_response[0]} (Ground Truth: {case['ground_truth']})"
            )
            print(
                f"   Correct: {'✅' if str(hybrid_response[0]) == case['ground_truth'] else '❌'}"
            )
        except Exception as e:
            print(f"   Error: {e}")

        print("\n" + "=" * 80)
        print("SUMMARY:")
        print(f"Reference Topic: {case['reference_contexts'][0]}")
        if topic_bm25_results:
            print(
                f"Classified Topic: {topic_bm25_results[0].get('classified_topic', 'Unknown')}"
            )
            print(
                f"Topic Match: {'✅' if topic_bm25_results[0].get('classified_topic') == case['reference_contexts'][0] else '❌'}"
            )
        print("=" * 80)


def analyze_random_sample():
    """Analyze a random sample of test cases"""
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, "data", "processed", "combined_train")

    print("Loading random sample of test cases...")
    test_cases = load_test_cases(data_dir, project_root, num_samples=5)

    print("Initializing diagnostic system...")
    llm_client = LocalLLMClient(model_name="cogito:14b")
    llm_client.ensure_model_available()
    diagnostic = BiobertRAG(llm_client=llm_client)
    smart_rag = SmartRAG(llm_client=llm_client)

    results = {
        "semantic_correct": 0,
        "bm25_correct": 0,
        "topic_bm25_correct": 0,
        "smart_rag_correct": 0,
        "hybrid_correct": 0,
        "total": len(test_cases),
    }

    for i, case in enumerate(test_cases):
        print(f"\n{'=' * 60}")
        print(f"CASE {i + 1}/{len(test_cases)}: {case['statement_id']}")
        print(f"{'=' * 60}")
        print(f"Question: {case['question'][:100]}...")
        print(f"Ground Truth: {case['ground_truth']}")
        print(f"Topic: {case['reference_contexts'][0]}")

        # Get results from all methods
        semantic_results = diagnostic.semantic_only_retrieval(case["question"], k=5)
        bm25_results = diagnostic.bm25_only_retrieval(case["question"], k=5)
        topic_bm25_results = diagnostic.topic_classified_bm25_retrieval(
            case["question"], k=5
        )
        hybrid_results = diagnostic.hybrid_retrieval(case["question"], k=5)

        # Get contexts
        semantic_context = "\n".join(
            [doc["document"]["text"] for doc in semantic_results]
        )
        bm25_context = "\n".join([doc["document"]["text"] for doc in bm25_results])
        topic_bm25_context = "\n".join(
            [doc["document"]["text"] for doc in topic_bm25_results]
        )
        hybrid_context = "\n".join([doc["document"]["text"] for doc in hybrid_results])

        # Get LLM responses
        try:
            semantic_response = llm_client.classify_statement(
                case["question"], semantic_context
            )
            semantic_correct = str(semantic_response[0]) == case["ground_truth"]
            if semantic_correct:
                results["semantic_correct"] += 1
        except Exception:
            semantic_correct = False

        try:
            bm25_response = llm_client.classify_statement(
                case["question"], bm25_context
            )
            bm25_correct = str(bm25_response[0]) == case["ground_truth"]
            if bm25_correct:
                results["bm25_correct"] += 1
        except Exception:
            bm25_correct = False

        try:
            topic_bm25_response = llm_client.classify_statement(
                case["question"], topic_bm25_context
            )
            topic_bm25_correct = str(topic_bm25_response[0]) == case["ground_truth"]
            if topic_bm25_correct:
                results["topic_bm25_correct"] += 1
        except Exception:
            topic_bm25_correct = False

        try:
            hybrid_response = llm_client.classify_statement(
                case["question"], hybrid_context
            )
            hybrid_correct = str(hybrid_response[0]) == case["ground_truth"]
            if hybrid_correct:
                results["hybrid_correct"] += 1
        except Exception:
            hybrid_correct = False

        # Test SmartRAG
        try:
            smart_rag_result = smart_rag.run(
                case["question"], case["reference_contexts"]
            )
            smart_rag_answer = json.loads(smart_rag_result["answer"])
            smart_rag_correct = (
                str(smart_rag_answer["statement_is_true"]) == case["ground_truth"]
            )
            if smart_rag_correct:
                results["smart_rag_correct"] += 1
        except Exception as e:
            print(f"SmartRAG error: {e}")
            smart_rag_correct = False

        # Show topic classification info
        if topic_bm25_results:
            classified_topic = topic_bm25_results[0].get("classified_topic", "Unknown")
            topic_match = classified_topic == case["reference_contexts"][0]
            print("\nTopic Classification:")
            print(f"  Reference: {case['reference_contexts'][0]}")
            print(f"  Classified: {classified_topic}")
            print(f"  Match: {'✅' if topic_match else '❌'}")

        print("\nResults:")
        print(f"  Semantic:     {'✅' if semantic_correct else '❌'}")
        print(f"  BM25:         {'✅' if bm25_correct else '❌'}")
        print(f"  Topic-BM25:   {'✅' if topic_bm25_correct else '❌'}")
        print(f"  SmartRAG:     {'✅' if smart_rag_correct else '❌'}")
        print(f"  Hybrid:       {'✅' if hybrid_correct else '❌'}")

        # Show problematic cases in detail (commented out to avoid memory issues)
        # if not hybrid_correct and bm25_correct:
        #     print(f"\n⚠️  HYBRID FAILED BUT BM25 SUCCEEDED - ANALYZING...")
        #     diagnostic.analyze_question(case["question"], case["ground_truth"], case["reference_contexts"])
        #     input("Press Enter to continue...")

    print(f"\n{'=' * 60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"Semantic Accuracy:     {results['semantic_correct']}/{results['total']} = {results['semantic_correct'] / results['total']:.2%}"
    )
    print(
        f"BM25 Accuracy:         {results['bm25_correct']}/{results['total']} = {results['bm25_correct'] / results['total']:.2%}"
    )
    print(
        f"Topic-BM25 Accuracy:   {results['topic_bm25_correct']}/{results['total']} = {results['topic_bm25_correct'] / results['total']:.2%}"
    )
    print(
        f"🚀 SmartRAG Accuracy:  {results['smart_rag_correct']}/{results['total']} = {results['smart_rag_correct'] / results['total']:.2%}"
    )
    print(
        f"Hybrid Accuracy:       {results['hybrid_correct']}/{results['total']} = {results['hybrid_correct'] / results['total']:.2%}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug BioBERT retrieval issues")
    parser.add_argument(
        "--mode",
        choices=["specific", "random"],
        default="specific",
        help="Analysis mode: specific examples or random sample",
    )

    args = parser.parse_args()

    if args.mode == "specific":
        analyze_specific_examples()
    else:
        analyze_random_sample()
