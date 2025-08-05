#!/usr/bin/env python3
"""
This script performs comprehensive analysis of the training data including:
- Basic statistics and counts
- Class balance analysis
- Topic distribution analysis
- Statement length analysis
- Data quality checks
"""

import json
import csv
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np

# Set paths relative to script location
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "eda"
TOPICS_FILE = DATA_DIR / "topics.json"

# Generation targets
TARGET_TOTAL = 5  # aim for ~4-5 total statements per topic
TARGET_PER_CLASS = 2  # at least 2 true AND 2 false


def load_topics() -> Dict[str, int]:
    """Load topics mapping from topics.json"""
    with open(TOPICS_FILE, "r") as f:
        return json.load(f)


def get_dataset_dirs(ds: str):
    if ds == "raw":
        base = DATA_DIR / "raw" / "train"
    elif ds == "combined":
        base = DATA_DIR / "processed" / "combined_train"
    else:
        raise ValueError("dataset must be 'raw' or 'combined'")
    return base / "statements", base / "answers"


def load_statements_and_answers(dataset: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Load all training statements and their corresponding answers for specified dataset"""
    statements = []
    answers = []

    statements_dir, answers_dir = get_dataset_dirs(dataset)

    # Get all statement files and sort them numerically
    statement_files = sorted(
        statements_dir.glob("statement_*.txt"), key=lambda x: int(x.stem.split("_")[1])
    )

    for stmt_file in statement_files:
        # Load statement text
        with open(stmt_file, "r", encoding="utf-8") as f:
            statement_text = f.read().strip()
            statements.append(statement_text)

        # Load corresponding answer
        answer_file = answers_dir / f"{stmt_file.stem}.json"
        with open(answer_file, "r") as f:
            answer_data = json.load(f)
            answers.append(answer_data)

    return statements, answers


def analyze_basic_stats(
    statements: List[str], answers: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate basic statistics about the dataset"""
    stats = {}

    # Basic counts
    stats["total_samples"] = len(statements)
    stats["total_answers"] = len(answers)

    # Verify data consistency
    stats["data_consistent"] = len(statements) == len(answers)

    # Statement lengths
    statement_lengths = [len(statement) for statement in statements]

    stats["statement_lengths"] = {
        "min": min(statement_lengths),
        "max": max(statement_lengths),
        "mean": np.mean(statement_lengths),
        "median": np.median(statement_lengths),
        "std": np.std(statement_lengths),
    }

    # Word counts
    word_counts = [len(statement.split()) for statement in statements]

    stats["word_counts"] = {
        "min": min(word_counts),
        "max": max(word_counts),
        "mean": np.mean(word_counts),
        "median": np.median(word_counts),
        "std": np.std(word_counts),
    }

    return stats


def analyze_class_balance(answers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the class balance for statement_is_true"""
    true_count = sum(1 for answer in answers if answer["statement_is_true"] == 1)
    false_count = len(answers) - true_count

    return {
        "true_statements": true_count,
        "false_statements": false_count,
        "true_percentage": (true_count / len(answers)) * 100,
        "false_percentage": (false_count / len(answers)) * 100,
        "balance_ratio": true_count / false_count if false_count > 0 else float("inf"),
    }


def analyze_topic_distribution(
    answers: List[Dict[str, Any]], topics: Dict[str, int]
) -> Dict[str, Any]:
    """Analyze the distribution of statement topics"""
    # Create reverse mapping: topic_id -> topic_name
    id_to_topic = {v: k for k, v in topics.items()}

    # Count topic occurrences
    topic_counts = Counter(answer["statement_topic"] for answer in answers)

    # Convert to topic names and sort by count
    topic_distribution = []
    for topic_id, count in topic_counts.most_common():
        topic_name = id_to_topic.get(topic_id, f"Unknown_{topic_id}")
        percentage = (count / len(answers)) * 100
        topic_distribution.append(
            {
                "topic_id": topic_id,
                "topic_name": topic_name,
                "count": count,
                "percentage": percentage,
            }
        )

    # Statistics
    counts = list(topic_counts.values())

    count_stats = {
        "min": min(counts),
        "max": max(counts),
        "mean": np.mean(counts),
        "median": np.median(counts),
        "std": np.std(counts),
    }

    stats = {
        "unique_topics": len(topic_counts),
        "total_possible_topics": len(topics),
        "coverage": (len(topic_counts) / len(topics)) * 100,
        "distribution": topic_distribution,
        "count_stats": count_stats,
    }

    return stats


def analyze_topic_truth_correlation(
    answers: List[Dict[str, Any]], topics: Dict[str, int]
) -> Dict[str, Any]:
    """Analyze correlation between topics and truth values"""
    id_to_topic = {v: k for k, v in topics.items()}

    # Group by topic
    topic_truth_stats = defaultdict(lambda: {"true": 0, "false": 0})

    for answer in answers:
        topic_id = answer["statement_topic"]
        is_true = answer["statement_is_true"]

        if is_true == 1:
            topic_truth_stats[topic_id]["true"] += 1
        else:
            topic_truth_stats[topic_id]["false"] += 1

    # Calculate percentages and ratios
    topic_analysis = []
    for topic_id, counts in topic_truth_stats.items():
        total = counts["true"] + counts["false"]
        topic_name = id_to_topic.get(topic_id, f"Unknown_{topic_id}")

        true_pct = (counts["true"] / total) * 100 if total > 0 else 0
        false_pct = (counts["false"] / total) * 100 if total > 0 else 0

        topic_analysis.append(
            {
                "topic_id": topic_id,
                "topic_name": topic_name,
                "total_statements": total,
                "true_count": counts["true"],
                "false_count": counts["false"],
                "true_percentage": true_pct,
                "false_percentage": false_pct,
            }
        )

    # Sort by total statements (most common topics first)
    topic_analysis.sort(key=lambda x: x["total_statements"], reverse=True)

    return {"topic_truth_analysis": topic_analysis}


def analyze_topic_gaps(
    answers: List[Dict[str, Any]], topics: Dict[str, int]
) -> Dict[str, List[Dict[str, Any]]]:
    """Analyze missing, sparse, and one-sided topics"""

    # Build comprehensive topic stats
    topic_stats = {}

    # Initialize all topics with zero counts
    for topic_name, topic_id in topics.items():
        topic_stats[topic_id] = {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "total_statements": 0,
            "true_count": 0,
            "false_count": 0,
        }

    # Count actual statements
    for answer in answers:
        topic_id = answer["statement_topic"]
        is_true = answer["statement_is_true"]

        if topic_id in topic_stats:
            topic_stats[topic_id]["total_statements"] += 1
            if is_true == 1:
                topic_stats[topic_id]["true_count"] += 1
            else:
                topic_stats[topic_id]["false_count"] += 1

    # Categorize topics
    missing_topics = []
    sparse_topics = []
    onesided_topics = []

    for topic_data in topic_stats.values():
        total = topic_data["total_statements"]
        true_count = topic_data["true_count"]
        false_count = topic_data["false_count"]

        if total == 0:
            missing_topics.append(topic_data)
        elif total > 0 and (true_count == 0 or false_count == 0):
            onesided_topics.append(topic_data)
        elif total < 4:
            sparse_topics.append(topic_data)

    # Sort each list by topic_id for consistency
    missing_topics.sort(key=lambda x: x["topic_id"])
    sparse_topics.sort(key=lambda x: x["topic_id"])
    onesided_topics.sort(key=lambda x: x["topic_id"])

    return {
        "missing_topics": missing_topics,
        "sparse_topics": sparse_topics,
        "onesided_topics": onesided_topics,
    }


def analyze_generation_needs(
    answers: List[Dict[str, Any]], topics: Dict[str, int]
) -> List[Dict[str, Any]]:
    """Calculate data generation needs for each topic to reach baseline targets"""

    # Build comprehensive topic stats
    topic_stats = {}

    # Initialize all topics with zero counts
    for topic_name, topic_id in topics.items():
        topic_stats[topic_id] = {
            "topic_id": topic_id,
            "topic_name": topic_name,
            "total_statements": 0,
            "true_count": 0,
            "false_count": 0,
        }

    # Count actual statements
    for answer in answers:
        topic_id = answer["statement_topic"]
        is_true = answer["statement_is_true"]

        if topic_id in topic_stats:
            topic_stats[topic_id]["total_statements"] += 1
            if is_true == 1:
                topic_stats[topic_id]["true_count"] += 1
            else:
                topic_stats[topic_id]["false_count"] += 1

    # Calculate generation needs for each topic
    generation_needs = []

    for topic_data in topic_stats.values():
        total = topic_data["total_statements"]
        true_count = topic_data["true_count"]
        false_count = topic_data["false_count"]

        # Calculate basic needs to meet per-class targets
        needed_true = max(0, TARGET_PER_CLASS - true_count)
        needed_false = max(0, TARGET_PER_CLASS - false_count)

        # If total statements < TARGET_TOTAL, ensure we reach the target
        current_after_additions = total + needed_true + needed_false
        if current_after_additions < TARGET_TOTAL:
            extra = TARGET_TOTAL - current_after_additions
            # Split extra evenly between true/false (round up for true)
            needed_true += math.ceil(extra / 2)
            needed_false += math.floor(extra / 2)

        generation_needs.append(
            {
                "topic_id": topic_data["topic_id"],
                "topic_name": topic_data["topic_name"],
                "total": total,
                "true_count": true_count,
                "false_count": false_count,
                "needed_true": needed_true,
                "needed_false": needed_false,
            }
        )

    # Sort by topic_id for consistency
    generation_needs.sort(key=lambda x: x["topic_id"])

    return generation_needs


def create_visualizations(
    stats: Dict[str, Any],
    topic_stats: Dict[str, Any],
    topic_gaps: Dict[str, List[Dict[str, Any]]],
    dataset: str = "raw",
) -> List[str]:
    """Create visualization plots and save them"""
    plt.style.use("default")
    saved_plots = []

    # Set up the plotting
    fig_dir = PROCESSED_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Load data once for all visualizations
    statements, answers = load_statements_and_answers(dataset)
    topics = load_topics()
    id_to_topic = {v: k for k, v in topics.items()}
    topic_counts = Counter(answer["statement_topic"] for answer in answers)

    # 1. Statement length distribution
    plt.figure(figsize=(10, 6))
    statement_lengths = [len(statement) for statement in statements]

    plt.hist(statement_lengths, bins=30, alpha=0.7, edgecolor="black")
    plt.title("Distribution of Statement Lengths (Characters)")
    plt.xlabel("Statement Length (Characters)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    plot_path = fig_dir / "statement_length_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots.append(str(plot_path))

    # 2. Word count distribution
    plt.figure(figsize=(10, 6))
    word_counts = [len(statement.split()) for statement in statements]

    plt.hist(word_counts, bins=30, alpha=0.7, edgecolor="black", color="orange")
    plt.title("Distribution of Statement Word Counts")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    plot_path = fig_dir / "word_count_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots.append(str(plot_path))

    # 3. Top 20 topics by frequency
    plt.figure(figsize=(12, 8))
    top_topics = topic_stats["distribution"][:20]

    topic_names = [item["topic_name"] for item in top_topics]
    counts = [item["count"] for item in top_topics]

    plt.barh(range(len(topic_names)), counts)
    plt.yticks(range(len(topic_names)), topic_names)
    plt.xlabel("Number of Statements")
    plt.title("Top 20 Topics by Frequency")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plot_path = fig_dir / "top_topics_frequency.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots.append(str(plot_path))

    # 4. Topic coverage visualization - all topics with categories highlighted
    plt.figure(figsize=(16, 32))

    # Collect complete topic data
    all_topics_data = []
    sparse_topic_ids = {t["topic_id"] for t in topic_gaps["sparse_topics"]}
    missing_topic_ids = {t["topic_id"] for t in topic_gaps["missing_topics"]}
    onesided_topic_ids = {t["topic_id"] for t in topic_gaps["onesided_topics"]}

    # Check all topics and define their category
    for topic_id, topic_name in id_to_topic.items():
        count = topic_counts.get(topic_id, 0)

        if topic_id in missing_topic_ids:
            category = "Missing (0 statements)"
            color = "red"
        elif topic_id in onesided_topic_ids:
            category = "One-sided (all true or false)"
            color = "purple"
        elif topic_id in sparse_topic_ids:
            category = "Sparse (<4 statements)"
            color = "orange"
        else:
            category = "Normal (≥4 balanced statements)"
            color = "steelblue"

        all_topics_data.append(
            {
                "name": topic_name,
                "count": count,
                "id": topic_id,
                "category": category,
                "color": color,
            }
        )

    # Sort by count (descending) to show most frequent topics first
    all_topics_data.sort(key=lambda x: x["count"], reverse=True)

    # Create the plot data
    names = [item["name"] for item in all_topics_data]
    counts = [item["count"] for item in all_topics_data]
    colors = [item["color"] for item in all_topics_data]

    # Make missing topics visible by giving them a small minimum width
    max_count = max(counts) if counts else 1
    min_visible_width = max_count * 0.015  # 1.5% of max count

    # Adjust counts for visualization (missing topics get minimum width)
    display_counts = [
        max(count, min_visible_width) if count == 0 else count for count in counts
    ]

    # Create horizontal bar chart
    y_positions = range(len(names))
    plt.barh(
        y_positions,
        display_counts,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )

    # Show ALL topic names
    ytick_positions = list(range(len(names)))
    ytick_labels = names
    plt.yticks(ytick_positions, ytick_labels, fontsize=6)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="red",
            alpha=0.7,
            label=f"Missing ({len(missing_topic_ids)} topics)",
        ),
        Patch(
            facecolor="purple",
            alpha=0.7,
            label=f"One-sided ({len(onesided_topic_ids)} topics)",
        ),
        Patch(
            facecolor="orange",
            alpha=0.7,
            label=f"Sparse ({len(sparse_topic_ids)} topics)",
        ),
        Patch(
            facecolor="steelblue",
            alpha=0.7,
            label=f"Normal ({115 - len(missing_topic_ids) - len(onesided_topic_ids) - len(sparse_topic_ids)} topics)",
        ),
    ]
    plt.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.xlabel("Number of Statements", fontsize=12)
    plt.ylabel("Topics (sorted by frequency)", fontsize=12)
    plt.title(
        "Topic Coverage Analysis - All 115 Topics\nSorted by Statement Count (Highest to Lowest)",
        fontsize=14,
        pad=20,
    )
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    plot_path = fig_dir / "topic_coverage.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots.append(str(plot_path))

    return saved_plots


def generate_markdown_report(
    stats: Dict[str, Any],
    class_balance: Dict[str, Any],
    topic_stats: Dict[str, Any],
    topic_truth: Dict[str, Any],
    topic_gaps: Dict[str, List[Dict[str, Any]]],
    generation_needs: List[Dict[str, Any]],
    plot_paths: List[str],
) -> str:
    """Generate comprehensive markdown report"""

    report = f"""# Emergency Healthcare RAG - Exploratory Data Analysis Report

**Generated on:** {Path(__file__).name} 
**Dataset:** Norwegian AI Championship 2025 - Emergency Healthcare RAG Track

## Executive Summary

This report provides a comprehensive analysis of the training dataset for the Emergency Healthcare RAG challenge. The dataset contains {stats["total_samples"]} medical statements with corresponding truth labels and topic classifications across {topic_stats["unique_topics"]} different medical topics.

## Dataset Overview

### Basic Statistics
- **Total Samples:** {stats["total_samples"]:,}
- **Data Consistency:** {"Consistent" if stats["data_consistent"] else " Inconsistent"}
- **Unique Topics Covered:** {topic_stats["unique_topics"]} out of {topic_stats["total_possible_topics"]} ({topic_stats["coverage"]:.1f}% coverage)

### Statement Characteristics

#### Length Analysis (Characters)
- **Minimum:** {stats["statement_lengths"]["min"]} characters
- **Maximum:** {stats["statement_lengths"]["max"]} characters  
- **Mean:** {stats["statement_lengths"]["mean"]:.1f} characters
- **Median:** {stats["statement_lengths"]["median"]:.1f} characters
- **Standard Deviation:** {stats["statement_lengths"]["std"]:.1f} characters

#### Word Count Analysis
- **Minimum:** {stats["word_counts"]["min"]} words
- **Maximum:** {stats["word_counts"]["max"]} words
- **Mean:** {stats["word_counts"]["mean"]:.1f} words
- **Median:** {stats["word_counts"]["median"]:.1f} words
- **Standard Deviation:** {stats["word_counts"]["std"]:.1f} words

## Class Balance Analysis

### Truth Value Distribution
- **True Statements:** {class_balance["true_statements"]:,} ({class_balance["true_percentage"]:.1f}%)
- **False Statements:** {class_balance["false_statements"]:,} ({class_balance["false_percentage"]:.1f}%)
- **Balance Ratio (True:False):** {class_balance["balance_ratio"]:.2f}:1

{"###  Class Imbalance Warning" if abs(class_balance["balance_ratio"] - 1.0) > 0.3 else "###  Reasonably Balanced Classes"}

{f"The dataset shows {'significant' if abs(class_balance['balance_ratio'] - 1.0) > 0.5 else 'moderate'} class imbalance. Consider stratified sampling and appropriate evaluation metrics." if abs(class_balance["balance_ratio"] - 1.0) > 0.3 else "The classes are reasonably balanced, which is good for model training and evaluation."}

## Topic Distribution Analysis

### Coverage Statistics
- **Topics with Data:** {topic_stats["unique_topics"]} topics
- **Topics without Data:** {topic_stats["total_possible_topics"] - topic_stats["unique_topics"]} topics
- **Coverage Percentage:** {topic_stats["coverage"]:.1f}%

### Topic Frequency Statistics
- **Most Common Topic:** {topic_stats["distribution"][0]["count"]} statements ({topic_stats["distribution"][0]["topic_name"]})
- **Least Common Topic:** {topic_stats["distribution"][-1]["count"]} statements ({topic_stats["distribution"][-1]["topic_name"]})
- **Average per Topic:** {topic_stats["count_stats"]["mean"]:.1f} statements
- **Median per Topic:** {topic_stats["count_stats"]["median"]:.1f} statements

### Top 10 Topics by Frequency

| Rank | Topic | Count | Percentage |
|------|--------|--------|-----------|"""

    # Add top 10 topics table
    for i, topic in enumerate(topic_stats["distribution"][:10], 1):
        report += f"\n| {i} | {topic['topic_name']} | {topic['count']} | {topic['percentage']:.1f}% |"

    report += """

### Bottom 10 Topics by Frequency

| Rank | Topic | Count | Percentage |
|------|--------|--------|-----------|"""

    # Add bottom 10 topics table
    bottom_topics = topic_stats["distribution"][-10:]
    for i, topic in enumerate(bottom_topics, 1):
        rank = len(topic_stats["distribution"]) - len(bottom_topics) + i
        report += f"\n| {rank} | {topic['topic_name']} | {topic['count']} | {topic['percentage']:.1f}% |"

    report += """

## Topic-Truth Correlation Analysis

### Topics with Highest True Statement Ratios

| Topic | Total | True | False | True % |
|--------|--------|--------|--------|---------|"""

    # Sort by true percentage for top true ratios
    sorted_by_true_pct = sorted(
        topic_truth["topic_truth_analysis"],
        key=lambda x: (x["true_percentage"], x["total_statements"]),
        reverse=True,
    )

    for topic in sorted_by_true_pct[:10]:
        report += f"\n| {topic['topic_name']} | {topic['total_statements']} | {topic['true_count']} | {topic['false_count']} | {topic['true_percentage']:.1f}% |"

    report += """

### Topics with Highest False Statement Ratios

| Topic | Total | True | False | False % |
|--------|--------|--------|--------|---------|"""

    # Sort by false percentage for top false ratios
    sorted_by_false_pct = sorted(
        topic_truth["topic_truth_analysis"],
        key=lambda x: (x["false_percentage"], x["total_statements"]),
        reverse=True,
    )

    for topic in sorted_by_false_pct[:10]:
        report += f"\n| {topic['topic_name']} | {topic['total_statements']} | {topic['true_count']} | {topic['false_count']} | {topic['false_percentage']:.1f}% |"

    # Add the three new tables
    report += f"""

## Topic Gap Analysis

### Missing Topics ({len(topic_gaps["missing_topics"])} topics)

Topics with zero training statements:

| Topic ID | Topic Name | Total | True | False |
|----------|------------|-------|------|-------|"""

    for topic in topic_gaps["missing_topics"]:
        report += f"\n| {topic['topic_id']} | {topic['topic_name']} | {topic['total_statements']} | {topic['true_count']} | {topic['false_count']} |"

    report += f"""

### Sparse Topics ({len(topic_gaps["sparse_topics"])} topics)

Topics with fewer than 4 training statements:

| Topic ID | Topic Name | Total | True | False |
|----------|------------|-------|------|-------|"""

    for topic in topic_gaps["sparse_topics"]:
        report += f"\n| {topic['topic_id']} | {topic['topic_name']} | {topic['total_statements']} | {topic['true_count']} | {topic['false_count']} |"

    report += f"""

### One-Sided Topics ({len(topic_gaps["onesided_topics"])} topics)

Topics where all statements are either true or false:

| Topic ID | Topic Name | Total | True | False |
|----------|------------|-------|------|-------|"""

    for topic in topic_gaps["onesided_topics"]:
        report += f"\n| {topic['topic_id']} | {topic['topic_name']} | {topic['total_statements']} | {topic['true_count']} | {topic['false_count']} |"

    report += """

## Data Quality Observations

### Potential Issues
"""

    # Identify potential issues
    issues = []

    if topic_stats["coverage"] < 100:
        missing_topics = (
            topic_stats["total_possible_topics"] - topic_stats["unique_topics"]
        )
        issues.append(
            f"- **Missing Topics:** {missing_topics} topics have no training data"
        )

    if abs(class_balance["balance_ratio"] - 1.0) > 0.5:
        issues.append(
            f"- **Severe Class Imbalance:** {class_balance['balance_ratio']:.2f}:1 ratio between true and false statements"
        )

    # Check for topics with very few samples
    low_sample_topics = [t for t in topic_stats["distribution"] if t["count"] <= 2]
    if low_sample_topics:
        issues.append(
            f"- **Low Sample Topics:** {len(low_sample_topics)} topics have ≤2 statements"
        )

    # Check for extreme statement lengths
    if stats["statement_lengths"]["max"] > 1000:
        issues.append(
            f"- **Very Long Statements:** Maximum length is {stats['statement_lengths']['max']} characters"
        )

    if stats["statement_lengths"]["min"] < 50:
        issues.append(
            f"- **Very Short Statements:** Minimum length is {stats['statement_lengths']['min']} characters"
        )

    if not issues:
        report += "-  No significant data quality issues identified"
    else:
        report += "\n".join(issues)

    # Add generation needs section
    topics_with_needs = [
        t for t in generation_needs if t["needed_true"] > 0 or t["needed_false"] > 0
    ]
    total_true_needed = sum(t["needed_true"] for t in generation_needs)
    total_false_needed = sum(t["needed_false"] for t in generation_needs)

    report += f"""

## Generation Needs

Full CSV at data/eda/topic_needs.csv

**Target:** {TARGET_TOTAL} total statements per topic, with at least {TARGET_PER_CLASS} true and {TARGET_PER_CLASS} false statements each.

**Summary:**
- Topics needing additional data: {len(topics_with_needs)} out of {len(generation_needs)}
- Total true statements to generate: {total_true_needed}
- Total false statements to generate: {total_false_needed}
- Total additional statements needed: {total_true_needed + total_false_needed}

### Preview: First 15 Topics with Generation Needs

| Topic ID | Topic Name | Needed True | Needed False |
|----------|------------|-------------|-------------|"""

    # Show first 15 topics with any needs
    preview_topics = topics_with_needs[:15]
    for topic in preview_topics:
        report += f"\n| {topic['topic_id']} | {topic['topic_name']} | {topic['needed_true']} | {topic['needed_false']} |"

    if len(topics_with_needs) > 15:
        report += f"\n\n*Showing first 15 of {len(topics_with_needs)} topics that need additional data.*"

    report += """



*This analysis was generated automatically by the EDA pipeline. For questions or issues, refer to the source code in `src/rag/eda.py`.*
"""

    return report


def main():
    """Main EDA execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="Run EDA script")
    parser.add_argument(
        "--dataset",
        type=str,
        default="raw",
        choices=["raw", "combined"],
        help="Dataset to analyze (default: raw)",
    )
    args = parser.parse_args()

    dataset = args.dataset

    print(f" Starting Emergency Healthcare RAG EDA on {dataset} dataset...")

    # Load data
    topics = load_topics()
    statements, answers = load_statements_and_answers(dataset)

    # Perform analyses
    basic_stats = analyze_basic_stats(statements, answers)
    class_balance = analyze_class_balance(answers)

    topic_stats = analyze_topic_distribution(answers, topics)
    topic_truth = analyze_topic_truth_correlation(answers, topics)
    topic_gaps = analyze_topic_gaps(answers, topics)

    # Analyze generation needs
    generation_needs = analyze_generation_needs(answers, topics)

    # Save generation needs to CSV
    needs_csv_path = PROCESSED_DIR / "topic_needs.csv"
    with open(needs_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "topic_id",
            "topic_name",
            "total",
            "true_count",
            "false_count",
            "needed_true",
            "needed_false",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(generation_needs)

    print(f"Generation needs analysis saved to: {needs_csv_path}")

    try:
        plot_paths = create_visualizations(
            basic_stats, topic_stats, topic_gaps, dataset
        )
        print(f" Created {len(plot_paths)} visualizations")
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")
        plot_paths = []

    # Determine report filename based on dataset
    if dataset == "combined":
        report_filename = "combined_eda_report.md"
    else:
        report_filename = "eda_report.md"

    report = generate_markdown_report(
        basic_stats,
        class_balance,
        topic_stats,
        topic_truth,
        topic_gaps,
        generation_needs,
        plot_paths,
    )

    # Save report
    report_path = PROCESSED_DIR / report_filename
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"EDA complete! Report saved to: {report_path}")

    # Print concise console summary
    print("\n EDA Summary:")
    print(f"  - Missing topics: {len(topic_gaps['missing_topics'])}")
    print(f"  - Sparse topics: {len(topic_gaps['sparse_topics'])}")
    print(f"  - One-sided topics: {len(topic_gaps['onesided_topics'])}")

    print("\n Generation plan:")
    topics_with_needs = [
        t for t in generation_needs if t["needed_true"] > 0 or t["needed_false"] > 0
    ]
    total_true_needed = sum(t["needed_true"] for t in generation_needs)
    total_false_needed = sum(t["needed_false"] for t in generation_needs)

    print(f"  - Topics needing any data : {len(topics_with_needs)}")
    print(f"  - Total true statements to generate : {total_true_needed}")
    print(f"  - Total false statements to generate: {total_false_needed}")

    print(
        "\nEDA complete! All analyses, visualizations, and reports have been generated."
    )


if __name__ == "__main__":
    main()
