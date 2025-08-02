#!/usr/bin/env python3
"""
Convert emergency healthcare competition data to RAG evaluation testset format.
"""

import json
import os
import glob
from typing import List, Dict


def load_training_data(data_dir: str) -> List[Dict]:
    """Load training statements and answers from the competition data."""
    statements_dir = os.path.join(data_dir, "train", "statements")
    answers_dir = os.path.join(data_dir, "train", "answers")

    testset = []

    # Find all statement files
    statement_files = glob.glob(os.path.join(statements_dir, "statement_*.txt"))

    for statement_file in sorted(statement_files):
        # Extract statement number from filename
        filename = os.path.basename(statement_file)
        statement_num = filename.replace("statement_", "").replace(".txt", "")

        # Read statement
        with open(statement_file, "r") as f:
            statement = f.read().strip()

        # Read corresponding answer
        answer_file = os.path.join(answers_dir, f"statement_{statement_num}.json")
        if os.path.exists(answer_file):
            with open(answer_file, "r") as f:
                answer_data = json.load(f)

            # Create testset entry
            testset_entry = {
                "question": statement,
                "ground_truth": f"Statement is {'true' if answer_data['statement_is_true'] else 'false'}, topic: {answer_data['statement_topic']}",
                "contexts": [statement],  # Using statement as context for now
            }

            testset.append(testset_entry)

    return testset


def load_topics_from_directory(topics_dir: str) -> List[str]:
    """Load all topic documents as contexts."""
    contexts = []

    for root, dirs, files in os.walk(topics_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.strip():  # Only add non-empty files
                        contexts.append(content)

    return contexts


def create_components_data(topics_dir: str) -> List[Dict]:
    """Create components data from topic files."""
    components = []

    for root, dirs, files in os.walk(topics_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, topics_dir)

                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                components.append(
                    {
                        "name": file.replace(".md", ""),
                        "path": relative_path,
                        "content": content[:500] + "..."
                        if len(content) > 500
                        else content,  # Truncate for overview
                    }
                )

    return components


def main():
    # Paths
    script_dir = os.path.dirname(__file__)
    data_root = "data"  # Points to the copied data directory

    # Load training data
    print("Loading training data...")
    testset = load_training_data(data_root)
    print(f"Loaded {len(testset)} training examples")

    # Load topic documents as additional contexts
    print("Loading topic documents...")
    topics_dir = os.path.join(data_root, "topics")
    topic_contexts = load_topics_from_directory(topics_dir)
    print(f"Loaded {len(topic_contexts)} topic documents")

    # Enhance testset with topic contexts
    for entry in testset:
        # Add some relevant topic contexts (for now, add a few random ones)
        # In a real implementation, you'd want to do semantic matching
        entry["contexts"].extend(topic_contexts[:5])  # Add first 5 contexts

    # Save testset
    testset_path = os.path.join(script_dir, "data", "datasets", "testset.json")
    os.makedirs(os.path.dirname(testset_path), exist_ok=True)

    with open(testset_path, "w") as f:
        json.dump(testset, f, indent=2)

    print(f"Saved testset to {testset_path}")

    # Create components data
    print("Creating components data...")
    components = create_components_data(topics_dir)

    components_path = os.path.join(script_dir, "data", "components.json")
    with open(components_path, "w") as f:
        json.dump(components, f, indent=2)

    print(f"Saved components to {components_path}")
    print(
        f"Created testset with {len(testset)} questions and {len(components)} components"
    )


if __name__ == "__main__":
    main()
