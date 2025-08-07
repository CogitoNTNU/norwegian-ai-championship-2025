import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime
import time
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from fact_checker import check_fact
from get_config import config


class DatasetTester:
    """Test the fact checker against labeled dataset."""
    
    def __init__(self):
        """
        Initialize the dataset tester using config.
        """
        self.statements_dir = Path(config.statements_dir)
        self.answers_dir = Path(config.answers_dir)
        self.topics_file = Path(config.topics_file)
        
        # Load topics mapping
        with open(self.topics_file, 'r') as f:
            self.topics = json.load(f)
        self.topic_id_to_name = {v: k for k, v in self.topics.items()}
        
        # Results storage
        self.results = []
        self.errors = []
        
    def load_dataset(self) -> List[Dict]:
        """
        Load the dataset of statements and their ground truth labels.
            
        Returns:
            List of dictionaries with statement, truth value, and topic
        """
        dataset = []
        
        # Get all statement files
        statement_files = sorted(self.statements_dir.glob("statement_*.txt"))
        
        for stmt_file in statement_files:
            # Extract the ID from filename
            file_id = stmt_file.stem  # e.g., "statement_0000"
            
            # Load statement
            with open(stmt_file, 'r', encoding='utf-8') as f:
                statement = f.read().strip()
            
            # Load corresponding answer
            answer_file = self.answers_dir / f"{file_id}.json"
            if not answer_file.exists():
                logger.warning(f"No answer file for {file_id}")
                continue
                
            with open(answer_file, 'r') as f:
                answer = json.load(f)
            
            # Map topic ID to name
            topic_id = answer["statement_topic"]
            topic_name = self.topic_id_to_name.get(topic_id, f"Unknown_{topic_id}")
            
            dataset.append({
                "id": file_id,
                "statement": statement,
                "ground_truth": bool(answer["statement_is_true"]),
                "ground_truth_topic_id": topic_id,
                "ground_truth_topic": topic_name
            })
        
        return dataset
    
    def test_single_statement(self, sample: Dict) -> Dict:
        """
        Test a single statement and compare with ground truth.
        
        Args:
            sample: Dictionary with statement and ground truth
            
        Returns:
            Dictionary with test results
        """
        start_time = time.time()
        
        try:
            # Get prediction from fact checker
            result = check_fact(sample["statement"])
            
            # Map verdict to boolean (TRUE -> True, FALSE -> False, UNVERIFIABLE -> None)
            if result["verdict"] == "TRUE":
                predicted = True
            elif result["verdict"] == "FALSE":
                predicted = False
            else:
                predicted = None
            
            # Compare with ground truth
            if predicted is None:
                is_correct = None  # Can't evaluate unverifiable
            else:
                is_correct = (predicted == sample["ground_truth"])
                if is_correct:
                    prediction_type = "CORRECT"
                else:
                    if sample["ground_truth"]:
                        prediction_type = "FALSE_NEGATIVE"  # Said FALSE but was TRUE
                    else:
                        prediction_type = "FALSE_POSITIVE"  # Said TRUE but was FALSE
            
            # Check topic match
            predicted_topic = result.get("topic", "unknown")
            topic_match = self.check_topic_match(predicted_topic, sample["ground_truth_topic"])
            
            elapsed_time = time.time() - start_time
            logger.info(f"Processing time: {elapsed_time:.2f}s")
            
            return {
                "id": sample["id"],
                "statement": sample["statement"][:100] + "..." if len(sample["statement"]) > 100 else sample["statement"],
                "ground_truth": sample["ground_truth"],
                "ground_truth_topic": sample["ground_truth_topic"],
                "predicted": predicted,
                "predicted_verdict": result["verdict"],
                "original_verdict": result.get("original_verdict", result["verdict"]),
                "predicted_topic": predicted_topic,
                "is_correct": is_correct,
                "prediction_type": prediction_type,
                "topic_match": topic_match,
                "chunks_retrieved": result.get("chunks_retrieved", 0),
                "avg_relevance": result.get("avg_relevance_score", 0),
                "time_seconds": elapsed_time,
                "error": None
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            return {
                "id": sample["id"],
                "statement": sample["statement"][:100],
                "ground_truth": sample["ground_truth"],
                "ground_truth_topic": sample["ground_truth_topic"],
                "predicted": None,
                "predicted_verdict": "ERROR",
                "predicted_topic": None,
                "is_correct": None,
                "prediction_type": "ERROR",
                "topic_match": False,
                "chunks_retrieved": 0,
                "avg_relevance": 0,
                "time_seconds": elapsed_time,
                "error": str(e)
            }
    
    def check_topic_match(self, predicted_topic: str, ground_truth_topic: str) -> bool:
        """
        Check if predicted topic matches ground truth (fuzzy matching).
        """
        if not predicted_topic or not ground_truth_topic:
            return False
        
        predicted_lower = predicted_topic.lower()
        truth_lower = ground_truth_topic.lower()
        
        # Exact match
        if predicted_lower == truth_lower:
            return True
        
        # Partial match (key words)
        truth_words = set(truth_lower.replace('_', ' ').replace('-', ' ').split())
        predicted_words = set(predicted_lower.replace('_', ' ').replace('-', ' ').split())
        
        # If significant overlap in words
        overlap = truth_words.intersection(predicted_words)
        if len(overlap) >= min(2, len(truth_words) // 2):
            return True
        
        return False
    
    def run_test(self) -> Dict:
        """
        Run the test on the entire dataset.
            
        Returns:
            Dictionary with test results and metrics
        """
        # Load dataset
        logger.info(f"Loading dataset from {self.statements_dir}")
        dataset = self.load_dataset()
        
        if not dataset:
            logger.error("No data found!")
            return {}
        
        logger.info(f"Loaded {len(dataset)} samples")
        logger.info(f"Using k={config.k} chunks, model={config.model_name}")
        logger.info("-" * 60)
        
        # Test each statement
        for i, sample in enumerate(dataset, 1):
            logger.info(f"Testing {i}/{len(dataset)}: {sample['id']}")
            
            result = self.test_single_statement(sample)
            self.results.append(result)
            
            if result["error"]:
                logger.error(f"ERROR: {result['error']}")
            else:
                correct_str = "✓ CORRECT" if result["is_correct"] else "✗ WRONG"
                topic_str = "✓ TOPIC MATCH" if result["topic_match"] else "✗ TOPIC MISMATCH"
                logger.info(f"{correct_str} ({result['predicted_verdict']}), {topic_str}")
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Always create plots and save results
        if config.plot_results:
            self.plot_results()
        if config.save_results:
            self.save_results()
        
        return metrics
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics from test results.
        """
        # Filter out unverifiable and errors for accuracy calculation
        valid_results = [r for r in self.results if r["is_correct"] is not None]
        
        if not valid_results:
            return {"error": "No valid results to calculate metrics"}
        
        # Basic metrics
        total = len(self.results)
        correct = sum(1 for r in valid_results if r["is_correct"])
        incorrect = sum(1 for r in valid_results if not r["is_correct"])
        errors = sum(1 for r in self.results if r["prediction_type"] == "ERROR")
        
        # Accuracy (only on verifiable predictions)
        accuracy = correct / len(valid_results) if valid_results else 0
        
        # True/False breakdown
        true_positives = sum(1 for r in valid_results if r["ground_truth"] and r["predicted"])
        true_negatives = sum(1 for r in valid_results if not r["ground_truth"] and not r["predicted"])
        false_positives = sum(1 for r in valid_results if not r["ground_truth"] and r["predicted"])
        false_negatives = sum(1 for r in valid_results if r["ground_truth"] and not r["predicted"])
        
        # Precision, Recall, F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Topic accuracy
        topic_matches = sum(1 for r in self.results if r["topic_match"])
        topic_accuracy = topic_matches / total if total > 0 else 0

        # Combined correctness: both verdict and topic are correct
        combined_correct = sum(
            1 for r in valid_results if r["is_correct"] and r["topic_match"]
        )
        combined_accuracy = combined_correct / len(valid_results) if valid_results else 0
        
        
        # Average processing time
        avg_time = np.mean([r["time_seconds"] for r in self.results])
        
        
        metrics = {
            "total_samples": total,
            "correct_predictions": correct,
            "incorrect_predictions": incorrect,
            "errors": errors,
            "accuracy": {
                "accuracy": accuracy,
                "topic_accuracy": topic_accuracy,
                "combined_accuracy": combined_accuracy,
            },
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": {
                "true_positives": true_positives,
                "true_negatives": true_negatives,
                "false_positives": false_positives,
                "false_negatives": false_negatives
            },
            "avg_processing_time": avg_time,
            "avg_chunks_retrieved": np.mean([r["chunks_retrieved"] for r in self.results]),
            "avg_relevance_score": np.mean([r["avg_relevance"] for r in self.results if r["avg_relevance"] > 0]),
        }
        
        return metrics
    
    def print_report(self, metrics: Dict):
        """
        Print a formatted report of test results using loguru.
        """
        logger.info("\n" + "="*70)
        logger.info("FACT CHECKER PERFORMANCE REPORT")
        logger.info("="*70)
        
        logger.info(f"\nOVERALL PERFORMANCE")
        logger.info(f"   Total Samples: {metrics['total_samples']}")
        logger.info(f"   Correct: {metrics['correct_predictions']} ({100*metrics['correct_predictions']/metrics['total_samples']:.1f}%)")
        logger.info(f"   Incorrect: {metrics['incorrect_predictions']} ({100*metrics['incorrect_predictions']/metrics['total_samples']:.1f}%)")
        logger.info(f"   Errors: {metrics['errors']} ({100*metrics['errors']/metrics['total_samples']:.1f}%)")
        
        logger.info(f"\nCLASSIFICATION METRICS")
        logger.info(f"   Accuracy: {metrics['accuracy']['accuracy']:.3f}")
        logger.info(f"   Precision: {metrics['precision']:.3f}")
        logger.info(f"   Recall: {metrics['recall']:.3f}")
        logger.info(f"   F1 Score: {metrics['f1_score']:.3f}")
        logger.info(f"   Topic Accuracy: {metrics['accuracy']['topic_accuracy']:.3f}")
        
        logger.info(f"\nCONFUSION MATRIX")
        cm = metrics['confusion_matrix']
        logger.info(f"   True Positives: {cm['true_positives']}")
        logger.info(f"   True Negatives: {cm['true_negatives']}")
        logger.info(f"   False Positives: {cm['false_positives']}")
        logger.info(f"   False Negatives: {cm['false_negatives']}")
        
        
        logger.info(f"\nPROCESSING STATS")
        logger.info(f"   Avg Time per Query: {metrics['avg_processing_time']:.2f}s")
        logger.info(f"   Avg Chunks Retrieved: {metrics['avg_chunks_retrieved']:.1f}")
        logger.info(f"   Avg Relevance Score: {metrics['avg_relevance_score']:.3f}")
        
    
    def save_results(self, output_dir: str = "test_results"):
        """
        Save detailed results to files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = output_path / f"detailed_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save metrics
        metrics = self.calculate_metrics()
        metrics_file = output_path / f"metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save errors separately
        errors = [r for r in self.results if r["error"]]
        if errors:
            errors_file = output_path / f"errors_{timestamp}.json"
            with open(errors_file, 'w') as f:
                json.dump(errors, f, indent=2)
        
        # Create CSV for easy analysis
        df = pd.DataFrame(self.results)
        csv_file = output_path / f"results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"\nResults saved to {output_path}")
        logger.info(f"   - Detailed results: {results_file.name}")
        logger.info(f"   - Metrics: {metrics_file.name}")
        logger.info(f"   - CSV: {csv_file.name}")
        if errors:
            logger.info(f"   - Errors: {errors_file.name}")
    
    def plot_results(self):
        """
        Create visualization plots of the results.
        """
        metrics = self.calculate_metrics()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Confusion Matrix
        cm = metrics['confusion_matrix']
        cm_array = np.array([[cm['true_negatives'], cm['false_positives']],
                            [cm['false_negatives'], cm['true_positives']]])
        
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted False', 'Predicted True'],
                   yticklabels=['Actual False', 'Actual True'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        
        # 2. Original Verdict Distribution (before UNSURE->TRUE mapping)
        original_verdict_counts = {}
        for r in self.results:
            original = r.get('original_verdict', r.get('predicted_verdict', 'UNKNOWN'))
            original_verdict_counts[original] = original_verdict_counts.get(original, 0) + 1
        
        if original_verdict_counts:
            axes[0, 1].pie(original_verdict_counts.values(), labels=original_verdict_counts.keys(), autopct='%1.1f%%')
            axes[0, 1].set_title('Original Verdict Distribution\n(Before UNSURE→TRUE Mapping)')
        
        # 3. Performance Metrics Bar Chart
        metrics_data = {
            'Accuracy': metrics['accuracy']['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score'],
            'Topic Acc': metrics['accuracy']['topic_accuracy']
        }
        axes[0, 2].bar(metrics_data.keys(), metrics_data.values(), color='skyblue')
        axes[0, 2].set_ylim([0, 1])
        axes[0, 2].set_title('Performance Metrics')
        axes[0, 2].set_ylabel('Score')
        
        # 4. Relevance Score Distribution
        relevance_scores = [r.get('avg_relevance', 0) for r in self.results if r.get('avg_relevance', 0) > 0]
        if relevance_scores:
            axes[1, 0].hist(relevance_scores, bins=15, color='lightgreen', alpha=0.7)
            axes[1, 0].set_title('Relevance Score Distribution')
            axes[1, 0].set_xlabel('Relevance Score')
            axes[1, 0].set_ylabel('Frequency')
        
        # 5. Result Types Distribution
        result_types = defaultdict(int)
        for r in self.results:
            result_types[r['prediction_type']] += 1
        
        axes[1, 1].bar(result_types.keys(), result_types.values(), color='coral')
        axes[1, 1].set_title('Prediction Types')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. UNSURE Mapping Impact
        unsure_count = sum(1 for r in self.results if r.get('original_verdict') == 'UNSURE')
        mapping_data = {
            'UNSURE→TRUE': unsure_count,
            'Direct TRUE': sum(1 for r in self.results if r.get('original_verdict') == 'TRUE'),
            'Direct FALSE': sum(1 for r in self.results if r.get('original_verdict') == 'FALSE')
        }
        axes[1, 2].bar(mapping_data.keys(), mapping_data.values(), color=['orange', 'green', 'red'])
        axes[1, 2].set_title('UNSURE Mapping Impact')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"test_results/performance_plot_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
        plt.show()


def main():
    """Main function using config parameters."""
    # Initialize tester with config parameters
    tester = DatasetTester()
    
    # Run test
    logger.info(f"Starting Medical Fact Checker Evaluation")
    logger.info(f"   Model: {config.model_name}")
    logger.info(f"   Chunks per query: {config.k}")
    logger.info("-" * 60)
    
    metrics = tester.run_test()
    
    # Print report
    if metrics:
        tester.print_report(metrics)
    
    # Results and plots are automatically saved due to config settings


if __name__ == "__main__":
    main()
