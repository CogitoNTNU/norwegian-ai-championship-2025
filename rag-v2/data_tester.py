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
import os

from fact_checker import check_fact, format_context_with_topics
from get_config import config
from langchain_chroma import Chroma
from embeddings import get_embeddings_func
from hybrid_search import hybrid_similarity_search_with_score


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
        with open(self.topics_file, "r") as f:
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
            with open(stmt_file, "r", encoding="utf-8") as f:
                statement = f.read().strip()

            # Load corresponding answer
            answer_file = self.answers_dir / f"{file_id}.json"
            if not answer_file.exists():
                logger.warning(f"No answer file for {file_id}")
                continue

            with open(answer_file, "r") as f:
                answer = json.load(f)

            # Map topic ID to name
            topic_id = answer["statement_topic"]
            topic_name = self.topic_id_to_name.get(topic_id, f"Unknown_{topic_id}")

            dataset.append(
                {
                    "id": file_id,
                    "statement": statement,
                    "ground_truth": bool(answer["statement_is_true"]),
                    "ground_truth_topic_id": topic_id,
                    "ground_truth_topic": topic_name,
                }
            )

        return dataset

    def test_single_statement(self, sample: Dict, model_name: str = None) -> Dict:
        """
        Test a single statement and compare with ground truth.

        Args:
            sample: Dictionary with statement and ground truth

        Returns:
            Dictionary with test results
        """
        start_time = time.time()

        try:
            logger.debug(f"Ground truth sample: {sample}")
            # Get debug context (what LLM actually sees)
            debug_context = self.get_debug_context(sample["statement"])

            # Get prediction from fact checker
            result = check_fact(sample["statement"], model_name)
            logger.debug(f"Result from check_fact {result}")

            # Map verdict to boolean (TRUE -> True, FALSE -> False, UNVERIFIABLE -> None)
            verdict = result.get("verdict", "TRUE")  # Default to TRUE if missing
            if verdict == "TRUE":
                verdict_predicted = True
            elif verdict == "FALSE":
                verdict_predicted = False
            else:
                verdict_predicted = True
            logger.debug(f"Verdict predicted: {verdict_predicted}")

            # Compare with ground truth
            is_correct = verdict_predicted == sample["ground_truth"]
            if is_correct:
                prediction_type = "CORRECT"
            else:
                if sample["ground_truth"]:
                    prediction_type = "FALSE_NEGATIVE"  # Said FALSE but was TRUE
                else:
                    prediction_type = "FALSE_POSITIVE"  # Said TRUE but was FALSE

            # Check topic match
            predicted_topic = result.get("topic", "unknown")
            topic_match = self.check_topic_match(
                predicted_topic, sample["ground_truth_topic"]
            )

            elapsed_time = time.time() - start_time
            logger.info(f"Processing time: {elapsed_time:.2f}s")

            # Create result dictionary first
            test_result = {
                "id": sample["id"],
                "statement": sample["statement"][:100] + "..."
                if len(sample["statement"]) > 100
                else sample["statement"],
                "ground_truth": sample["ground_truth"],
                "ground_truth_topic": sample["ground_truth_topic"],
                "predicted": verdict_predicted,
                "predicted_verdict": verdict,
                "original_verdict": result.get("original_verdict", verdict),
                "predicted_topic": predicted_topic,
                "is_correct": is_correct,
                "prediction_type": prediction_type,
                "topic_match": topic_match,
                "chunks_retrieved": result.get("chunks_retrieved", 0),
                "avg_relevance": result.get("avg_relevance_score", 0),
                "time_seconds": elapsed_time,
                "error": None,
            }

            # Save debug info for wrong predictions
            if is_correct is not None and not is_correct:
                # Add full statement to result for debugging
                debug_result = test_result.copy()
                debug_result["predicted"] = verdict_predicted
                debug_result["topic_match"] = topic_match
                debug_result["raw_result"] = result  # Pass the raw result for debug
                self.save_wrong_prediction_debug(sample, debug_result, debug_context)

            return test_result

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
                "error": str(e),
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
        truth_words = set(truth_lower.replace("_", " ").replace("-", " ").split())
        predicted_words = set(
            predicted_lower.replace("_", " ").replace("-", " ").split()
        )

        # If significant overlap in words
        overlap = truth_words.intersection(predicted_words)
        if len(overlap) >= min(2, len(truth_words) // 2):
            return True

        return False

    def get_debug_context(self, statement: str) -> Dict:
        """
        Get the same context that the LLM sees for debugging purposes.

        Args:
            statement: The statement to check

        Returns:
            Dictionary with context chunks and formatted context
        """
        # Use hybrid or vector search based on configuration (same as in check_fact)
        if config.use_hybrid_search:
            results = hybrid_similarity_search_with_score(statement, k=config.k, bm25_weight=config.bm25_weight)
        else:
            # Fall back to pure vector search
            db = Chroma(
                persist_directory=config.chroma_path,
                embedding_function=get_embeddings_func(),
            )
            chroma_results = db.similarity_search_with_score(statement, k=config.k)
            # Convert to hybrid search format for consistency
            results = [(doc, 1-score) for doc, score in chroma_results]

        if not results:
            return {
                "chunks_retrieved": 0,
                "raw_chunks": [],
                "formatted_context": "No relevant chunks found",
                "chunk_scores": [],
            }

        # Format context the same way as sent to LLM
        formatted_context = format_context_with_topics(results)

        # Extract raw chunk information
        raw_chunks = []
        chunk_scores = []

        for i, (doc, score) in enumerate(results):
            chunk_info = {
                "chunk_index": i,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": float(score),  # Relevance score (higher = better)
                "search_method": "hybrid" if config.use_hybrid_search else "vector",
                "topic": doc.metadata.get("topic", "unknown"),
            }
            raw_chunks.append(chunk_info)
            chunk_scores.append(float(score))

        return {
            "chunks_retrieved": len(results),
            "raw_chunks": raw_chunks,
            "formatted_context": formatted_context,
            "chunk_scores": chunk_scores,
            "avg_relevance_score": sum(chunk_scores) / len(chunk_scores)
            if chunk_scores
            else 0,
        }

    def save_wrong_prediction_debug(
        self, sample: Dict, result: Dict, debug_context: Dict
    ):
        """
        Save debug information for wrong predictions.

        Args:
            sample: Ground truth sample information
            result: Model prediction result
            debug_context: Context retrieved from ChromaDB
        """
        # Create debug directory if it doesn't exist
        debug_dir = Path("data/wrong_pred")
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Create comprehensive debug information
        debug_info = {
            "sample_id": sample["id"],
            "timestamp": datetime.now().isoformat(),
            # Statement and ground truth
            "statement": sample["statement"],
            "ground_truth": {
                "is_true": sample["ground_truth"],
                "topic_id": sample["ground_truth_topic_id"],
                "topic_name": sample["ground_truth_topic"],
            },
            # Model predictions
            "predictions": {
                "verdict": result.get("raw_result", {}).get("verdict", result.get("predicted_verdict", "TRUE")),
                "original_verdict": result.get("raw_result", {}).get("original_verdict", result.get("predicted_verdict", "TRUE")),
                "predicted_topic": result.get("predicted_topic", "unknown"),
                "predicted_boolean": result.get("predicted", None),
            },
            # Error analysis
            "error_analysis": {
                "prediction_type": result.get("prediction_type", "UNKNOWN"),
                "is_correct": result.get("is_correct", None),
                "topic_match": result.get("topic_match", False),
                "processing_time_seconds": result.get("time_seconds", 0),
            },
            # ChromaDB context (what the LLM actually saw)
            "llm_context": {
                "chunks_retrieved": debug_context["chunks_retrieved"],
                "avg_relevance_score": debug_context["avg_relevance_score"],
                "formatted_context_sent_to_llm": debug_context["formatted_context"],
                "individual_chunks": debug_context["raw_chunks"],
            },
            # Model configuration
            "model_config": {
                "k_chunks": config.k,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
            },
        }

        # Save to file named by sample ID and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{sample['id']}_{timestamp}_debug.json"
        filepath = debug_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(debug_info, f, indent=2, ensure_ascii=False)

        logger.info(f"Wrong prediction debug info saved: {filepath}")

    def run_test(self, model_name: str) -> Dict:
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

        # Apply statement range filtering if configured
        if config.from_statement is not None or config.to_statement is not None:
            from_idx = (config.from_statement or 1) - 1  # Convert to 0-based index
            to_idx = (
                config.to_statement if config.to_statement is not None else len(dataset)
            )

            # Ensure valid range
            from_idx = max(0, min(from_idx, len(dataset)))
            to_idx = max(from_idx, min(to_idx, len(dataset)))

            original_size = len(dataset)
            dataset = dataset[from_idx:to_idx]
            logger.info(
                f"Filtered dataset: statements {from_idx + 1}-{to_idx} ({len(dataset)}/{original_size} samples)"
            )
        else:
            logger.info(f"Testing all {len(dataset)} samples")

        current_model = model_name or config.model_names[0]
        logger.info(f"Using k={config.k} chunks, model={current_model}")
        logger.info("-" * 60)

        # Test each statement
        for i, sample in enumerate(dataset, 1):
            logger.info(f"Testing {i}/{len(dataset)}: {sample['id']}")

            result = self.test_single_statement(sample, current_model)
            self.results.append(result)

            if result["error"]:
                logger.error(f"ERROR: {result['error']}")
            correct_str = "✓ CORRECT" if result["is_correct"] else "✗ WRONG"
            topic_str = (
                "✓ TOPIC MATCH" if result["topic_match"] else "✗ TOPIC MISMATCH"
            )
            
            # Calculate running combined accuracy
            valid_results = [r for r in self.results if r["is_correct"] is not None]
            if valid_results:
                correctness_acc = sum(1 for r in valid_results if r["is_correct"]) / len(valid_results)
                topic_acc = sum(1 for r in self.results if r["topic_match"]) / len(self.results)
                combined_acc = (correctness_acc + topic_acc) / 2
                
                logger.info(
                    f"{correct_str} ({result['predicted_verdict']}), {topic_str} | Combined Accuracy: {combined_acc:.3f}"
                )
            else:
                logger.info(
                    f"{correct_str} ({result['predicted_verdict']}), {topic_str}"
                )

        # Calculate metrics
        metrics = self.calculate_metrics()

        # Always create plots and save results
        model_dir = f"test_results/{current_model.replace(':', '_')}"
        if config.plot_results:
            self.plot_results(current_model, model_dir)
        if config.save_results:
            self.save_results(model_dir)

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
        true_positives = sum(
            1 for r in valid_results if r["ground_truth"] and r["predicted"]
        )
        true_negatives = sum(
            1 for r in valid_results if not r["ground_truth"] and not r["predicted"]
        )
        false_positives = sum(
            1 for r in valid_results if not r["ground_truth"] and r["predicted"]
        )
        false_negatives = sum(
            1 for r in valid_results if r["ground_truth"] and not r["predicted"]
        )

        # Precision, Recall, F1
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Topic accuracy
        topic_matches = sum(1 for r in self.results if r["topic_match"])
        topic_accuracy = topic_matches / total if total > 0 else 0

        # Combined correctness: both verdict and topic are correct
        combined_correct = sum(
            1 for r in valid_results if r["is_correct"] and r["topic_match"]
        )
        combined_accuracy = (
            combined_correct / len(valid_results) if valid_results else 0
        )

        # Average processing time
        avg_time = np.mean([r["time_seconds"] for r in self.results])

        metrics = {
            "total_samples": total,
            "correct_predictions": correct,
            "incorrect_predictions": incorrect,
            "errors": errors,
            "accuracy": {
                "verdict_accuracy": accuracy,
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
                "false_negatives": false_negatives,
            },
            "avg_processing_time": avg_time,
            "avg_chunks_retrieved": np.mean(
                [r["chunks_retrieved"] for r in self.results]
            ),
            "avg_relevance_score": np.mean(
                [r["avg_relevance"] for r in self.results if r["avg_relevance"] > 0]
            ),
        }

        return metrics

    def print_report(self, metrics: Dict):
        """
        Print a formatted report of test results using loguru.
        """
        logger.info("\n" + "=" * 70)
        logger.info("FACT CHECKER PERFORMANCE REPORT")
        logger.info("=" * 70)

        logger.info(f"\nOVERALL PERFORMANCE")
        logger.info(f"   Total Samples: {metrics['total_samples']}")
        logger.info(
            f"   Correct: {metrics['correct_predictions']} ({100 * metrics['correct_predictions'] / metrics['total_samples']:.1f}%)"
        )
        logger.info(
            f"   Incorrect: {metrics['incorrect_predictions']} ({100 * metrics['incorrect_predictions'] / metrics['total_samples']:.1f}%)"
        )
        logger.info(
            f"   Errors: {metrics['errors']} ({100 * metrics['errors'] / metrics['total_samples']:.1f}%)"
        )

        logger.info(f"\nCLASSIFICATION METRICS")
        logger.info(f"   Accuracy: {metrics['accuracy']['verdict_accuracy']:.3f}")
        logger.info(f"   Precision: {metrics['precision']:.3f}")
        logger.info(f"   Recall: {metrics['recall']:.3f}")
        logger.info(f"   F1 Score: {metrics['f1_score']:.3f}")
        logger.info(f"   Topic Accuracy: {metrics['accuracy']['topic_accuracy']:.3f}")

        logger.info(f"\nCONFUSION MATRIX")
        cm = metrics["confusion_matrix"]
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
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # Save metrics
        metrics = self.calculate_metrics()
        metrics_file = output_path / f"metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save errors separately
        errors = [r for r in self.results if r["error"]]
        if errors:
            errors_file = output_path / f"errors_{timestamp}.json"
            with open(errors_file, "w") as f:
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

    def plot_results(self, model_name: str = None, output_dir: str = "test_results"):
        """
        Create visualization plots of the results.
        """
        metrics = self.calculate_metrics()

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Confusion Matrix
        cm = metrics["confusion_matrix"]
        cm_array = np.array(
            [
                [cm["true_negatives"], cm["false_positives"]],
                [cm["false_negatives"], cm["true_positives"]],
            ]
        )

        sns.heatmap(
            cm_array,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted False", "Predicted True"],
            yticklabels=["Actual False", "Actual True"],
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Confusion Matrix")

        # 2. Original Verdict Distribution (before UNSURE->TRUE mapping)
        original_verdict_counts = {}
        for r in self.results:
            original = r.get("original_verdict", r.get("predicted_verdict", "UNKNOWN"))
            original_verdict_counts[original] = (
                original_verdict_counts.get(original, 0) + 1
            )

        if original_verdict_counts:
            axes[0, 1].pie(
                original_verdict_counts.values(),
                labels=original_verdict_counts.keys(),
                autopct="%1.1f%%",
            )
            axes[0, 1].set_title(
                "Original Verdict Distribution\n(Before UNSURE→TRUE Mapping)"
            )

        # 3. Performance Metrics Bar Chart
        metrics_data = {
            "Accuracy": metrics["accuracy"]["verdict_accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1_score"],
            "Topic Acc": metrics["accuracy"]["topic_accuracy"],
        }
        axes[0, 2].bar(metrics_data.keys(), metrics_data.values(), color="skyblue")
        axes[0, 2].set_ylim([0, 1])
        axes[0, 2].set_title("Performance Metrics")
        axes[0, 2].set_ylabel("Score")

        # 4. Relevance Score Distribution
        relevance_scores = [
            r.get("avg_relevance", 0)
            for r in self.results
            if r.get("avg_relevance", 0) > 0
        ]
        if relevance_scores:
            axes[1, 0].hist(relevance_scores, bins=15, color="lightgreen", alpha=0.7)
            axes[1, 0].set_title("Relevance Score Distribution")
            axes[1, 0].set_xlabel("Relevance Score")
            axes[1, 0].set_ylabel("Frequency")

        # 5. Result Types Distribution
        result_types = defaultdict(int)
        for r in self.results:
            result_types[r["prediction_type"]] += 1

        axes[1, 1].bar(result_types.keys(), result_types.values(), color="coral")
        axes[1, 1].set_title("Prediction Types")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].tick_params(axis="x", rotation=45)

        # 6. UNSURE Mapping Impact
        unsure_count = sum(
            1 for r in self.results if r.get("original_verdict") == "UNSURE"
        )
        mapping_data = {
            "UNSURE→TRUE": unsure_count,
            "Direct TRUE": sum(
                1 for r in self.results if r.get("original_verdict") == "TRUE"
            ),
            "Direct FALSE": sum(
                1 for r in self.results if r.get("original_verdict") == "FALSE"
            ),
        }
        axes[1, 2].bar(
            mapping_data.keys(), mapping_data.values(), color=["orange", "green", "red"]
        )
        axes[1, 2].set_title("UNSURE Mapping Impact")
        axes[1, 2].set_ylabel("Count")
        axes[1, 2].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_suffix = f"_{model_name.replace(':', '_')}" if model_name else ""
        save_path = f"{output_dir}/performance_plot{model_suffix}_{timestamp}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")
        plt.close()


def main():
    """Main function using config parameters."""
    logger.info("Starting Medical Fact Checker Evaluation")
    logger.info(f"   Models to test: {config.model_names}")
    logger.info(f"   Chunks per query: {config.k}")
    logger.info("=" * 80)

    for i, model_name in enumerate(config.model_names, 1):
        logger.info(f"\n[{i}/{len(config.model_names)}] Testing model: {model_name}")
        logger.info("-" * 60)

        # Initialize fresh tester for each model
        tester = DatasetTester()

        # Run test for this model
        start_time = time.time()
        metrics = tester.run_test(model_name)
        total_time = time.time() - start_time
        logger.info(f"Total time used on model {model_name} is {total_time}")

        # Print report
        if metrics:
            logger.info(f"\nResults for {model_name}:")
            tester.print_report(metrics)

        logger.info(f"Completed testing {model_name}")
        logger.info("=" * 80)

    logger.info(f"\nCompleted testing all {len(config.model_names)} models!")


if __name__ == "__main__":
    main()
