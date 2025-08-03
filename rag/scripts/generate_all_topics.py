#!/usr/bin/env python3
"""
Batch topic generation script for Emergency Healthcare RAG dataset.
Reads from topic_needs.csv and generates synthetic data for all topics that need it.
"""

import csv
import sys
import os
from typing import List, Dict

# Import the single topic generation function
from generate_topic import generate_topic_data


def load_topic_needs(csv_path: str) -> List[Dict]:
    """Load topic needs from CSV file."""
    topics_to_generate = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic_id = int(row["topic_id"])
            needed_true = int(row["needed_true"])
            needed_false = int(row["needed_false"])

            # Only include topics that need generation
            if needed_true > 0 or needed_false > 0:
                topics_to_generate.append(
                    {
                        "topic_id": topic_id,
                        "topic_name": row["topic_name"],
                        "needed_true": needed_true,
                        "needed_false": needed_false,
                        "total_needed": needed_true + needed_false,
                    }
                )

    return topics_to_generate


def generate_all_topics(
    csv_path: str, skip_errors: bool = True, start_from: int = None
) -> None:
    """Generate synthetic data for all topics that need it."""
    print(f"Loading topic needs from: {csv_path}")

    topics_to_generate = load_topic_needs(csv_path)

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

    total_topics = len(topics_to_generate)
    total_statements_needed = sum(t["total_needed"] for t in topics_to_generate)

    print(f"Found {total_topics} topics needing data generation")
    print(f"Total statements to generate: {total_statements_needed}")
    print("-" * 60)

    success_count = 0
    error_count = 0
    total_generated = 0

    for i, topic in enumerate(topics_to_generate, 1):
        topic_id = topic["topic_id"]
        topic_name = topic["topic_name"]
        needed_true = topic["needed_true"]
        needed_false = topic["needed_false"]

        print(f"\n[{i}/{total_topics}] Processing Topic {topic_id}: {topic_name}")
        print(f"Need: {needed_true} true, {needed_false} false statements")

        try:
            # Use the existing generation function
            generate_topic_data(topic_id, needed_true, needed_false)
            success_count += 1
            total_generated += needed_true + needed_false

        except Exception as e:
            error_count += 1
            print(f" ERROR generating Topic {topic_id}: {e}")

            if not skip_errors:
                print("Stopping due to error (use --skip-errors to continue)")
                sys.exit(1)
            else:
                print("Continuing to next topic...")

    # Final summary
    print("\n" + "=" * 60)
    print("BATCH GENERATION COMPLETE")
    print("=" * 60)
    print(f" Successfully generated: {success_count}/{total_topics} topics")
    print(f" Errors encountered: {error_count}/{total_topics} topics")
    print(f" Total statements generated: {total_generated}")

    if error_count > 0:
        print(f"\n  {error_count} topics had errors - check logs above")
    else:
        print("\n All topics generated successfully!")


def main():
    """Command-line interface for batch generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic data for all topics from CSV"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/eda/topic_needs.csv",
        help="Path to topic needs CSV file",
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

    args = parser.parse_args()

    # Validate CSV file exists
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        return 1

    try:
        generate_all_topics(args.csv_path, args.skip_errors, args.start_from)
        return 0
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
