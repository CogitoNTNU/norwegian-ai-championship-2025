#!/usr/bin/env python3
"""
Balanced data generation script for Emergency Healthcare RAG dataset.
Reads from synthetic_generation_needs.json and generates synthetic data to balance topics.
"""

import json
import sys
import os
from typing import List, Dict

# Import the single topic generation function
from generate_topic import generate_topic_data


def load_generation_needs(json_path: str) -> List[Dict]:
    """Load generation needs from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        all_needs = json.load(f)

    # Filter to only topics that need generation
    topics_to_generate = []
    for need in all_needs:
        needed_true = need["needed_true"]
        needed_false = need["needed_false"]

        if needed_true > 0 or needed_false > 0:
            topics_to_generate.append(
                {
                    "topic_id": need["topic_id"],
                    "topic_name": need["topic_name"],
                    "needed_true": needed_true,
                    "needed_false": needed_false,
                    "total_needed": needed_true + needed_false,
                    "current_total": need["current_total"],
                    "current_true": need["current_true"],
                    "current_false": need["current_false"],
                    "reason": need["reason"],
                }
            )

    return topics_to_generate


def generate_balanced_data(
    json_path: str,
    skip_errors: bool = True,
    start_from: int = None,
    max_topics: int = None,
) -> None:
    """Generate synthetic data for all topics that need balancing."""
    print(f"Loading generation needs from: {json_path}")

    topics_to_generate = load_generation_needs(json_path)

    # Filter topics to start from specific topic ID if requested
    if start_from is not None:
        original_count = len(topics_to_generate)
        topics_to_generate = [
            t for t in topics_to_generate if t["topic_id"] >= start_from
        ]
        skipped_count = original_count - len(topics_to_generate)
        print(
            f"Starting from topic {start_from} - skipping {skipped_count} earlier topics"
        )

    # Limit number of topics if requested
    if max_topics is not None:
        topics_to_generate = topics_to_generate[:max_topics]
        print(f"Limited to first {max_topics} topics")

    # Sort by total needed (prioritize topics needing most data)
    topics_to_generate.sort(key=lambda x: x["total_needed"], reverse=True)

    total_topics = len(topics_to_generate)
    total_statements_needed = sum(t["total_needed"] for t in topics_to_generate)

    print("\n=== Balanced Data Generation Plan ===")
    print(f"Topics needing data: {total_topics}")
    print(f"Total statements to generate: {total_statements_needed}")
    print(
        f"Breakdown: {sum(t['needed_true'] for t in topics_to_generate)} true, {sum(t['needed_false'] for t in topics_to_generate)} false"
    )
    print("-" * 60)

    success_count = 0
    error_count = 0
    total_generated = 0

    for i, topic in enumerate(topics_to_generate, 1):
        topic_id = topic["topic_id"]
        topic_name = topic["topic_name"]
        needed_true = topic["needed_true"]
        needed_false = topic["needed_false"]
        current_total = topic["current_total"]
        current_true = topic["current_true"]
        current_false = topic["current_false"]
        reason = topic["reason"]

        print(f"\n[{i}/{total_topics}] Topic {topic_id}: {topic_name}")
        print(
            f"  Current: {current_total} total ({current_true} true, {current_false} false)"
        )
        print(f"  Need: {needed_true} true, {needed_false} false")
        print(f"  Reason: {reason}")

        try:
            # Use the existing generation function
            if needed_true > 0 or needed_false > 0:
                generate_topic_data(topic_id, needed_true, needed_false)
                success_count += 1
                total_generated += needed_true + needed_false
            else:
                print("  Skipping - no data needed")
                success_count += 1

        except Exception as e:
            error_count += 1
            print(f"  ERROR generating Topic {topic_id}: {e}")

            if not skip_errors:
                print("Stopping due to error (use --skip-errors to continue)")
                sys.exit(1)
            else:
                print("  Continuing to next topic...")

    # Final summary
    print("\n" + "=" * 60)
    print("BALANCED DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Successfully processed: {success_count}/{total_topics} topics")
    print(f"Errors encountered: {error_count}/{total_topics} topics")
    print(f"Total statements generated: {total_generated}")

    if error_count > 0:
        print(f"\n{error_count} topics had errors - check logs above")
    else:
        print("\nAll topics processed successfully!")


def main():
    """Command-line interface for balanced data generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic data to balance topics based on needs analysis"
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="../data/processed/synthetic_generation_needs.json",
        help="Path to synthetic generation needs JSON file",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue processing if individual topics fail",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        help="Start generation from specific topic ID (resume from this topic)",
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        help="Maximum number of topics to process (useful for testing)",
    )

    args = parser.parse_args()

    # Validate JSON file exists
    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found: {args.json_path}")
        print(
            "Run the synthetic needs analysis first: python data/eda/generate_synthetic_needs.py"
        )
        return 1

    try:
        generate_balanced_data(
            args.json_path, args.skip_errors, args.start_from, args.max_topics
        )
        return 0
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
