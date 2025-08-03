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
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import numpy as np

# Set paths relative to script location
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "eda"
TOPICS_FILE = DATA_DIR / "topics.json"

def load_topics() -> Dict[str, int]:
    """Load topics mapping from topics.json"""
    with open(TOPICS_FILE, 'r') as f:
        return json.load(f)

def load_statements_and_answers() -> Tuple[List[str], List[Dict[str, Any]]]:
    """Load all training statements and their corresponding answers"""
    statements = []
    answers = []
    
    statements_dir = RAW_DIR / "train" / "statements"
    answers_dir = RAW_DIR / "train" / "answers"
    
    # Get all statement files and sort them numerically
    statement_files = sorted(statements_dir.glob("statement_*.txt"), 
                           key=lambda x: int(x.stem.split('_')[1]))
    
    for stmt_file in statement_files:
        # Load statement text
        with open(stmt_file, 'r', encoding='utf-8') as f:
            statement_text = f.read().strip()
            statements.append(statement_text)
        
        # Load corresponding answer
        answer_file = answers_dir / f"{stmt_file.stem}.json"
        with open(answer_file, 'r') as f:
            answer_data = json.load(f)
            answers.append(answer_data)
    
    return statements, answers

def analyze_basic_stats(statements: List[str], answers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate basic statistics about the dataset"""
    stats = {}
    
    # Basic counts
    stats['total_samples'] = len(statements)
    stats['total_answers'] = len(answers)
    
    # Verify data consistency
    stats['data_consistent'] = len(statements) == len(answers)
    
    # Statement lengths
    statement_lengths = [len(statement) for statement in statements]
    stats['statement_lengths'] = {
        'min': min(statement_lengths),
        'max': max(statement_lengths),
        'mean': np.mean(statement_lengths),
        'median': np.median(statement_lengths),
        'std': np.std(statement_lengths)
    }
    
    # Word counts
    word_counts = [len(statement.split()) for statement in statements]
    stats['word_counts'] = {
        'min': min(word_counts),
        'max': max(word_counts),
        'mean': np.mean(word_counts),
        'median': np.median(word_counts),
        'std': np.std(word_counts)
    }
    
    return stats

def analyze_class_balance(answers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the class balance for statement_is_true"""
    true_count = sum(1 for answer in answers if answer['statement_is_true'] == 1)
    false_count = len(answers) - true_count
    
    return {
        'true_statements': true_count,
        'false_statements': false_count,
        'true_percentage': (true_count / len(answers)) * 100,
        'false_percentage': (false_count / len(answers)) * 100,
        'balance_ratio': true_count / false_count if false_count > 0 else float('inf')
    }

def analyze_topic_distribution(answers: List[Dict[str, Any]], topics: Dict[str, int]) -> Dict[str, Any]:
    """Analyze the distribution of statement topics"""
    # Create reverse mapping: topic_id -> topic_name
    id_to_topic = {v: k for k, v in topics.items()}
    
    # Count topic occurrences
    topic_counts = Counter(answer['statement_topic'] for answer in answers)
    
    # Convert to topic names and sort by count
    topic_distribution = []
    for topic_id, count in topic_counts.most_common():
        topic_name = id_to_topic.get(topic_id, f"Unknown_{topic_id}")
        percentage = (count / len(answers)) * 100
        topic_distribution.append({
            'topic_id': topic_id,
            'topic_name': topic_name,
            'count': count,
            'percentage': percentage
        })
    
    # Statistics
    counts = list(topic_counts.values())
    stats = {
        'unique_topics': len(topic_counts),
        'total_possible_topics': len(topics),
        'coverage': (len(topic_counts) / len(topics)) * 100,
        'distribution': topic_distribution,
        'count_stats': {
            'min': min(counts),
            'max': max(counts),
            'mean': np.mean(counts),
            'median': np.median(counts),
            'std': np.std(counts)
        }
    }
    
    return stats

def analyze_topic_truth_correlation(answers: List[Dict[str, Any]], topics: Dict[str, int]) -> Dict[str, Any]:
    """Analyze correlation between topics and truth values"""
    id_to_topic = {v: k for k, v in topics.items()}
    
    # Group by topic
    topic_truth_stats = defaultdict(lambda: {'true': 0, 'false': 0})
    
    for answer in answers:
        topic_id = answer['statement_topic']
        is_true = answer['statement_is_true']
        
        if is_true == 1:
            topic_truth_stats[topic_id]['true'] += 1
        else:
            topic_truth_stats[topic_id]['false'] += 1
    
    # Calculate percentages and ratios
    topic_analysis = []
    for topic_id, counts in topic_truth_stats.items():
        total = counts['true'] + counts['false']
        topic_name = id_to_topic.get(topic_id, f"Unknown_{topic_id}")
        
        true_pct = (counts['true'] / total) * 100 if total > 0 else 0
        false_pct = (counts['false'] / total) * 100 if total > 0 else 0
        
        topic_analysis.append({
            'topic_id': topic_id,
            'topic_name': topic_name,
            'total_statements': total,
            'true_count': counts['true'],
            'false_count': counts['false'],
            'true_percentage': true_pct,
            'false_percentage': false_pct
        })
    
    # Sort by total statements (most common topics first)
    topic_analysis.sort(key=lambda x: x['total_statements'], reverse=True)
    
    return {'topic_truth_analysis': topic_analysis}

def create_visualizations(stats: Dict[str, Any], topic_stats: Dict[str, Any]) -> List[str]:
    """Create visualization plots and save them"""
    plt.style.use('default')
    saved_plots = []
    
    # Set up the plotting
    fig_dir = PROCESSED_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # 1. Statement length distribution
    plt.figure(figsize=(10, 6))
    # We'll need to reload the data to get actual lengths for histogram
    statements, _ = load_statements_and_answers()
    statement_lengths = [len(statement) for statement in statements]
    
    plt.hist(statement_lengths, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Statement Lengths (Characters)')
    plt.xlabel('Statement Length (Characters)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plot_path = fig_dir / "statement_length_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_plots.append(str(plot_path))
    
    # 2. Word count distribution
    plt.figure(figsize=(10, 6))
    word_counts = [len(statement.split()) for statement in statements]
    
    plt.hist(word_counts, bins=30, alpha=0.7, edgecolor='black', color='orange')
    plt.title('Distribution of Statement Word Counts')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plot_path = fig_dir / "word_count_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_plots.append(str(plot_path))
    
    # 3. Top 20 topics by frequency
    plt.figure(figsize=(12, 8))
    top_topics = topic_stats['distribution'][:20]
    
    topic_names = [item['topic_name'] for item in top_topics]
    counts = [item['count'] for item in top_topics]
    
    plt.barh(range(len(topic_names)), counts)
    plt.yticks(range(len(topic_names)), topic_names)
    plt.xlabel('Number of Statements')
    plt.title('Top 20 Topics by Frequency')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_path = fig_dir / "top_topics_frequency.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_plots.append(str(plot_path))
    
    return saved_plots

def generate_markdown_report(stats: Dict[str, Any], class_balance: Dict[str, Any], 
                           topic_stats: Dict[str, Any], topic_truth: Dict[str, Any],
                           plot_paths: List[str]) -> str:
    """Generate comprehensive markdown report"""
    
    report = f"""# Emergency Healthcare RAG - Exploratory Data Analysis Report

**Generated on:** {Path(__file__).name} 
**Dataset:** Norwegian AI Championship 2025 - Emergency Healthcare RAG Track

## Executive Summary

This report provides a comprehensive analysis of the training dataset for the Emergency Healthcare RAG challenge. The dataset contains {stats['total_samples']} medical statements with corresponding truth labels and topic classifications across {topic_stats['unique_topics']} different medical topics.

## Dataset Overview

### Basic Statistics
- **Total Samples:** {stats['total_samples']:,}
- **Data Consistency:** {'Consistent' if stats['data_consistent'] else ' Inconsistent'}
- **Unique Topics Covered:** {topic_stats['unique_topics']} out of {topic_stats['total_possible_topics']} ({topic_stats['coverage']:.1f}% coverage)

### Statement Characteristics

#### Length Analysis (Characters)
- **Minimum:** {stats['statement_lengths']['min']} characters
- **Maximum:** {stats['statement_lengths']['max']} characters  
- **Mean:** {stats['statement_lengths']['mean']:.1f} characters
- **Median:** {stats['statement_lengths']['median']:.1f} characters
- **Standard Deviation:** {stats['statement_lengths']['std']:.1f} characters

#### Word Count Analysis
- **Minimum:** {stats['word_counts']['min']} words
- **Maximum:** {stats['word_counts']['max']} words
- **Mean:** {stats['word_counts']['mean']:.1f} words
- **Median:** {stats['word_counts']['median']:.1f} words
- **Standard Deviation:** {stats['word_counts']['std']:.1f} words

## Class Balance Analysis

### Truth Value Distribution
- **True Statements:** {class_balance['true_statements']:,} ({class_balance['true_percentage']:.1f}%)
- **False Statements:** {class_balance['false_statements']:,} ({class_balance['false_percentage']:.1f}%)
- **Balance Ratio (True:False):** {class_balance['balance_ratio']:.2f}:1

{'###  Class Imbalance Warning' if abs(class_balance['balance_ratio'] - 1.0) > 0.3 else '###  Reasonably Balanced Classes'}

{f"The dataset shows {'significant' if abs(class_balance['balance_ratio'] - 1.0) > 0.5 else 'moderate'} class imbalance. Consider stratified sampling and appropriate evaluation metrics." if abs(class_balance['balance_ratio'] - 1.0) > 0.3 else "The classes are reasonably balanced, which is good for model training and evaluation."}

## Topic Distribution Analysis

### Coverage Statistics
- **Topics with Data:** {topic_stats['unique_topics']} topics
- **Topics without Data:** {topic_stats['total_possible_topics'] - topic_stats['unique_topics']} topics
- **Coverage Percentage:** {topic_stats['coverage']:.1f}%

### Topic Frequency Statistics
- **Most Common Topic:** {topic_stats['distribution'][0]['count']} statements ({topic_stats['distribution'][0]['topic_name']})
- **Least Common Topic:** {topic_stats['distribution'][-1]['count']} statements ({topic_stats['distribution'][-1]['topic_name']})
- **Average per Topic:** {topic_stats['count_stats']['mean']:.1f} statements
- **Median per Topic:** {topic_stats['count_stats']['median']:.1f} statements

### Top 10 Topics by Frequency

| Rank | Topic | Count | Percentage |
|------|--------|--------|-----------|"""

    # Add top 10 topics table
    for i, topic in enumerate(topic_stats['distribution'][:10], 1):
        report += f"\n| {i} | {topic['topic_name']} | {topic['count']} | {topic['percentage']:.1f}% |"

    report += f"""

### Bottom 10 Topics by Frequency

| Rank | Topic | Count | Percentage |
|------|--------|--------|-----------|"""

    # Add bottom 10 topics table
    bottom_topics = topic_stats['distribution'][-10:]
    for i, topic in enumerate(bottom_topics, 1):
        rank = len(topic_stats['distribution']) - len(bottom_topics) + i
        report += f"\n| {rank} | {topic['topic_name']} | {topic['count']} | {topic['percentage']:.1f}% |"

    report += f"""

## Topic-Truth Correlation Analysis

### Topics with Highest True Statement Ratios

| Topic | Total | True | False | True % |
|--------|--------|--------|--------|---------|"""

    # Sort by true percentage for top true ratios
    sorted_by_true_pct = sorted(topic_truth['topic_truth_analysis'], 
                               key=lambda x: (x['true_percentage'], x['total_statements']), 
                               reverse=True)
    
    for topic in sorted_by_true_pct[:10]:
        report += f"\n| {topic['topic_name']} | {topic['total_statements']} | {topic['true_count']} | {topic['false_count']} | {topic['true_percentage']:.1f}% |"

    report += f"""

### Topics with Highest False Statement Ratios

| Topic | Total | True | False | False % |
|--------|--------|--------|--------|---------|"""

    # Sort by false percentage for top false ratios  
    sorted_by_false_pct = sorted(topic_truth['topic_truth_analysis'], 
                                key=lambda x: (x['false_percentage'], x['total_statements']), 
                                reverse=True)
    
    for topic in sorted_by_false_pct[:10]:
        report += f"\n| {topic['topic_name']} | {topic['total_statements']} | {topic['true_count']} | {topic['false_count']} | {topic['false_percentage']:.1f}% |"

    report += """

## Data Quality Observations

### Potential Issues
"""

    # Identify potential issues
    issues = []
    
    if topic_stats['coverage'] < 100:
        missing_topics = topic_stats['total_possible_topics'] - topic_stats['unique_topics']
        issues.append(f"- **Missing Topics:** {missing_topics} topics have no training data")
    
    if abs(class_balance['balance_ratio'] - 1.0) > 0.5:
        issues.append(f"- **Severe Class Imbalance:** {class_balance['balance_ratio']:.2f}:1 ratio between true and false statements")
    
    # Check for topics with very few samples
    low_sample_topics = [t for t in topic_stats['distribution'] if t['count'] <= 2]
    if low_sample_topics:
        issues.append(f"- **Low Sample Topics:** {len(low_sample_topics)} topics have ≤2 statements")
    
    # Check for extreme statement lengths
    if stats['statement_lengths']['max'] > 1000:
        issues.append(f"- **Very Long Statements:** Maximum length is {stats['statement_lengths']['max']} characters")
    
    if stats['statement_lengths']['min'] < 50:
        issues.append(f"- **Very Short Statements:** Minimum length is {stats['statement_lengths']['min']} characters")
    
    if not issues:
        report += "-  No significant data quality issues identified"
    else:
        report += "\n".join(issues)

    report += f"""



*This analysis was generated automatically by the EDA pipeline. For questions or issues, refer to the source code in `src/rag/eda.py`.*
"""

    return report

def main():
    """Main EDA execution function"""
    print(" Starting Emergency Healthcare RAG EDA...")
    
    # Load data)
    topics = load_topics()
    statements, answers = load_statements_and_answers()
    
    
    # Perform analyses
    basic_stats = analyze_basic_stats(statements, answers)
    class_balance = analyze_class_balance(answers)
    
    topic_stats = analyze_topic_distribution(answers, topics)
    topic_truth = analyze_topic_truth_correlation(answers, topics)

    try:
        plot_paths = create_visualizations(basic_stats, topic_stats)
        print(f" Created {len(plot_paths)} visualizations")
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")
        plot_paths = []

    report = generate_markdown_report(basic_stats, class_balance, topic_stats, topic_truth, plot_paths)
    
    # Save report
    report_path = PROCESSED_DIR / "eda_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"EDA complete! Report saved to: {report_path}")

if __name__ == "__main__":
    main()