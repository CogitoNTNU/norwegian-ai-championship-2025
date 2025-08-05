#!/usr/bin/env python3
"""
Split the combined training data into 85% for BM25s indexing and 15% for testing.
This ensures proper evaluation without data leakage.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple


def load_combined_data() -> List[Tuple[str, str, dict]]:
    """Load all combined training data."""
    combined_dir = Path("data/processed/combined")
    statements_dir = combined_dir / "statements"
    answers_dir = combined_dir / "answers"

    if not statements_dir.exists() or not answers_dir.exists():
        raise FileNotFoundError(
            "Combined training data not found. Run combine_datasets.py first."
        )

    data = []
    for stmt_file in sorted(statements_dir.glob("statement_*.txt")):
        stmt_id = stmt_file.stem  # e.g., "statement_0001"
        answer_file = answers_dir / f"{stmt_id}.json"

        if answer_file.exists():
            # Read statement
            with open(stmt_file, "r", encoding="utf-8") as f:
                statement = f.read().strip()

            # Read answer
            with open(answer_file, "r", encoding="utf-8") as f:
                answer = json.load(f)

            data.append((stmt_id, statement, answer))

    return data


def split_data(
    data: List[Tuple[str, str, dict]], train_ratio: float = 0.85
) -> Tuple[List, List]:
    """Split data into training and test sets."""
    # Shuffle data deterministically for reproducible splits
    random.seed(42)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # Calculate split index
    split_idx = int(len(shuffled_data) * train_ratio)

    train_data = shuffled_data[:split_idx]
    test_data = shuffled_data[split_idx:]

    return train_data, test_data


def save_split_data(train_data: List, test_data: List):
    """Save split data to separate directories."""
    # Create directories
    train_dir = Path("data/processed/train")
    test_dir = Path("data/processed/test")

    for directory in [train_dir, test_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "statements").mkdir(exist_ok=True)
        (directory / "answers").mkdir(exist_ok=True)

    # Save training data
    print(f"Saving {len(train_data)} training samples...")
    for stmt_id, statement, answer in train_data:
        # Save statement
        stmt_file = train_dir / "statements" / f"{stmt_id}.txt"
        with open(stmt_file, "w", encoding="utf-8") as f:
            f.write(statement)

        # Save answer
        answer_file = train_dir / "answers" / f"{stmt_id}.json"
        with open(answer_file, "w", encoding="utf-8") as f:
            json.dump(answer, f, indent=2)

    # Save test data
    print(f"Saving {len(test_data)} test samples...")
    for stmt_id, statement, answer in test_data:
        # Save statement
        stmt_file = test_dir / "statements" / f"{stmt_id}.txt"
        with open(stmt_file, "w", encoding="utf-8") as f:
            f.write(statement)

        # Save answer
        answer_file = test_dir / "answers" / f"{stmt_id}.json"
        with open(answer_file, "w", encoding="utf-8") as f:
            json.dump(answer, f, indent=2)


def analyze_split(train_data: List, test_data: List):
    """Analyze the topic and truth value distribution in the split."""

    def get_stats(data, name):
        total = len(data)
        true_count = sum(1 for _, _, answer in data if answer["statement_is_true"] == 1)
        false_count = total - true_count

        # Count topics
        topics = {}
        for _, _, answer in data:
            topic = answer["statement_topic"]
            topics[topic] = topics.get(topic, 0) + 1

        print(f"\n{name} Set Statistics:")
        print(f"  Total samples: {total}")
        print(f"  True statements: {true_count} ({true_count / total * 100:.1f}%)")
        print(f"  False statements: {false_count} ({false_count / total * 100:.1f}%)")
        print(f"  Unique topics: {len(topics)}")

        return topics

    train_topics = get_stats(train_data, "Training")
    test_topics = get_stats(test_data, "Test")

    # Check topic overlap
    common_topics = set(train_topics.keys()) & set(test_topics.keys())
    print(f"\n  Topics in both sets: {len(common_topics)}")
    print(
        f"  Topics only in training: {len(set(train_topics.keys()) - set(test_topics.keys()))}"
    )
    print(
        f"  Topics only in test: {len(set(test_topics.keys()) - set(train_topics.keys()))}"
    )


def main():
    """Main function to split the data."""
    print("🔀 Splitting combined training data for BM25s indexing...")

    # Load combined data
    print("📁 Loading combined training data...")
    data = load_combined_data()
    print(f"Loaded {len(data)} samples")

    # Split data (85% training, 15% testing)
    print("✂️ Splitting data (85% train, 15% test)...")
    train_data, test_data = split_data(data, train_ratio=0.85)

    # Analyze split
    analyze_split(train_data, test_data)

    # Save split data
    print("\n💾 Saving split data...")
    save_split_data(train_data, test_data)

    print("\n✅ Data split complete!")
    print(f"  Training data: data/processed/train/ ({len(train_data)} samples)")
    print(f"  Test data: data/processed/test/ ({len(test_data)} samples)")
    print("  All methods will use only the training portion (85%)")


if __name__ == "__main__":
    main()
