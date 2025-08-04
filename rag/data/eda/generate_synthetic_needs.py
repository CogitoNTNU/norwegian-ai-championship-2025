#!/usr/bin/env python3
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

# Define paths relative to script location
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent

topics_file = DATA_DIR / "topics.json"


def load_topics() -> Dict[str, int]:
    with open(topics_file, "r") as f:
        return json.load(f)


# Function to load answers in JSON format
def load_answers() -> List[Dict[str, Any]]:
    answers = []
    answers_dir = DATA_DIR / "processed" / "combined_train" / "answers"
    for answer_file in answers_dir.glob("*.json"):
        with open(answer_file, "r") as f:
            answers.append(json.load(f))
    return answers


# Function to analyze generation needs
def analyze_generation_needs(
    answers: List[Dict[str, Any]], topics: Dict[str, int]
) -> List[Dict[str, Any]]:
    # Create reverse mapping: topic_id -> topic_name
    id_to_topic = {v: k for k, v in topics.items()}

    topic_stats = defaultdict(lambda: {"total": 0, "true": 0, "false": 0})

    for answer in answers:
        topic_id = answer["statement_topic"]
        is_true = answer["statement_is_true"]

        topic_stats[topic_id]["total"] += 1
        if is_true:
            topic_stats[topic_id]["true"] += 1
        else:
            topic_stats[topic_id]["false"] += 1

    generation_needs = []

    for topic_id, stats in topic_stats.items():
        topic_name = id_to_topic.get(topic_id, f"Unknown_{topic_id}")

        # Balance true/false statements
        needed_true = max(0, stats["false"] - stats["true"])
        needed_false = max(0, stats["true"] - stats["false"])

        # Ensure minimum of 4 statements per topic
        if stats["total"] < 4:
            extra_needed = 4 - stats["total"]
            needed_true += math.ceil(extra_needed / 2)
            needed_false += math.floor(extra_needed / 2)

        generation_needs.append(
            {
                "topic_id": topic_id,
                "topic_name": topic_name,
                "current_total": stats["total"],
                "current_true": stats["true"],
                "current_false": stats["false"],
                "needed_true": needed_true,
                "needed_false": needed_false,
                "reason": get_reason(stats, needed_true, needed_false),
            }
        )

    # Also include missing topics (topics with 0 statements)
    for topic_name, topic_id in topics.items():
        if topic_id not in topic_stats:
            generation_needs.append(
                {
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "current_total": 0,
                    "current_true": 0,
                    "current_false": 0,
                    "needed_true": 2,
                    "needed_false": 2,
                    "reason": "Missing topic (no statements)",
                }
            )

    # Sort by topic_id for consistency
    generation_needs.sort(key=lambda x: x["topic_id"])
    return generation_needs


def get_reason(stats: Dict[str, int], needed_true: int, needed_false: int) -> str:
    """Generate a reason for why synthetic data is needed"""
    total = stats["total"]
    true_count = stats["true"]
    false_count = stats["false"]

    reasons = []

    if total < 4:
        reasons.append(f"Sparse topic ({total} statements < 4 minimum)")

    if needed_true > 0 and needed_false > 0:
        reasons.append(f"Unbalanced (needs {needed_true} true, {needed_false} false)")
    elif needed_true > 0:
        reasons.append(f"Needs {needed_true} more true statements for balance")
    elif needed_false > 0:
        reasons.append(f"Needs {needed_false} more false statements for balance")

    if true_count == 0:
        reasons.append("One-sided (no true statements)")
    elif false_count == 0:
        reasons.append("One-sided (no false statements)")

    return "; ".join(reasons) if reasons else "Balanced and sufficient"


def main():
    topics = load_topics()
    answers = load_answers()
    needs = analyze_generation_needs(answers, topics)

    # Calculate summary statistics
    topics_needing_data = [
        n for n in needs if n["needed_true"] > 0 or n["needed_false"] > 0
    ]
    total_true_needed = sum(n["needed_true"] for n in needs)
    total_false_needed = sum(n["needed_false"] for n in needs)
    total_statements_needed = total_true_needed + total_false_needed

    missing_topics = [n for n in needs if n["current_total"] == 0]
    sparse_topics = [n for n in needs if 0 < n["current_total"] < 4]
    unbalanced_topics = [
        n
        for n in needs
        if n["current_total"] >= 4 and (n["needed_true"] > 0 or n["needed_false"] > 0)
    ]

    # Print summary
    print("\n=== Synthetic Data Generation Analysis ===")
    print(f"Total topics: {len(topics)}")
    print(f"Topics needing synthetic data: {len(topics_needing_data)}")
    print(
        f"Total statements to generate: {total_statements_needed} ({total_true_needed} true, {total_false_needed} false)"
    )
    print()
    print(f"Missing topics (0 statements): {len(missing_topics)}")
    print(f"Sparse topics (<4 statements): {len(sparse_topics)}")
    print(f"Unbalanced topics (≥4 but imbalanced): {len(unbalanced_topics)}")
    print()

    # Show top 10 topics needing most data
    top_needs = sorted(
        topics_needing_data,
        key=lambda x: x["needed_true"] + x["needed_false"],
        reverse=True,
    )[:10]
    if top_needs:
        print("Top 10 topics needing most synthetic data:")
        for i, topic in enumerate(top_needs, 1):
            total_needed = topic["needed_true"] + topic["needed_false"]
            print(
                f"{i:2d}. {topic['topic_name'][:40]:40} | Need: {total_needed:2d} ({topic['needed_true']} true, {topic['needed_false']} false) | Current: {topic['current_total']} ({topic['current_true']} true, {topic['current_false']} false)"
            )
        print()

    # Save the generation needs
    needs_file = DATA_DIR / "processed" / "synthetic_generation_needs.json"
    with open(needs_file, "w") as f:
        json.dump(needs, f, indent=2)

    print(f"Synthetic generation needs saved to: {needs_file}")


if __name__ == "__main__":
    main()
