#!/usr/bin/env python3
"""
Create evaluation testset from the BM25s held-out test data.
This ensures the evaluation uses the proper 15% test split.
"""

import json
import os
from pathlib import Path
from typing import Dict


def load_topics_mapping() -> Dict[int, str]:
    """Load the topic ID to name mapping."""
    topics_paths = ["data/topics.json", "../data/topics.json", "../../data/topics.json"]

    for path in topics_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                topics = json.load(f)
                # Reverse mapping: id -> name
                return {v: k for k, v in topics.items()}

    raise FileNotFoundError("Could not find topics.json")


def create_evaluation_testset():
    """Create evaluation testset from BM25s test data."""
    print("📝 Creating evaluation testset from BM25s test data...")

    # Load topic mapping
    id_to_topic = load_topics_mapping()

    # Load test data
    test_dir = Path("data/processed/test")
    statements_dir = test_dir / "statements"
    answers_dir = test_dir / "answers"

    if not statements_dir.exists() or not answers_dir.exists():
        raise FileNotFoundError("Test data not found. Run split_data_bm25.py first.")

    testset = []

    for stmt_file in sorted(statements_dir.glob("statement_*.txt")):
        stmt_id = stmt_file.stem
        answer_file = answers_dir / f"{stmt_id}.json"

        if answer_file.exists():
            # Read statement
            with open(stmt_file, "r", encoding="utf-8") as f:
                statement = f.read().strip()

            # Read answer
            with open(answer_file, "r", encoding="utf-8") as f:
                answer = json.load(f)

            # Get topic name from ID
            topic_id = answer.get("statement_topic", 0)
            topic_name = id_to_topic.get(topic_id, "Unknown")

            # Create testset entry
            testset_entry = {
                "user_input": statement,
                "reference": str(answer.get("statement_is_true", 1)),
                "reference_contexts": [topic_name],
            }

            testset.append(testset_entry)

    # Save testset to evaluation directory
    eval_dir = Path("rag-evaluation/data/processed/test")
    eval_dir.mkdir(parents=True, exist_ok=True)

    testset_path = eval_dir / "testset.json"
    with open(testset_path, "w", encoding="utf-8") as f:
        json.dump(testset, f, indent=2, ensure_ascii=False)

    print(f"✅ Created evaluation testset with {len(testset)} samples")
    print(f"   Saved to: {testset_path}")

    # Also create a direct symlink/copy in the expected location
    expected_path = Path("rag-evaluation/data/datasets/testset_bm25_split.json")
    expected_path.parent.mkdir(parents=True, exist_ok=True)

    with open(expected_path, "w", encoding="utf-8") as f:
        json.dump(testset, f, indent=2, ensure_ascii=False)

    print(f"   Also saved to: {expected_path}")

    # Show sample
    print("\n📋 Sample testset entries:")
    for i, entry in enumerate(testset[:3]):
        print(f"   {i + 1}. Statement: {entry['user_input'][:60]}...")
        print(
            f"      Reference: {entry['reference']} (topic: {entry['reference_contexts'][0]})"
        )

    return testset_path


def main():
    """Main function."""
    print("🧪 BM25s Evaluation Testset Creator")
    print("=" * 40)

    try:
        create_evaluation_testset()
        print("\n✅ Evaluation testset created successfully!")
        print("   The rag-evaluation system will now use the proper 15% test split")
        print("   No data leakage: evaluation uses only held-out test data")

    except Exception as e:
        print(f"❌ Error creating testset: {e}")
        return False

    return True


if __name__ == "__main__":
    main()
