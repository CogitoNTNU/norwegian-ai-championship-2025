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

# Import the fact checker
from fact_checker import check_fact


class DatasetTester:
    """Test the fact checker against labeled dataset."""
    
    def __init__(self, 
                 statements_dir: str = "data/train/statements",
                 answers_dir: str = "data/train/answers",
                 topics_file: str = "data/topics.json",
                 k: int = 20,
                 model_name: str = "cogito:32b"):
        """
        Initialize the dataset tester.
        
        Args:
            statements_dir: Directory containing statement txt files
            answers_dir: Directory containing answer json files
            topics_file: JSON file with topic mappings
            k: Number of chunks to retrieve
            model_name: Ollama model to use for fact checking (default: cogito:8b)
        """
        self.statements_dir = Path(statements_dir)
        self.answers_dir = Path(answers_dir)
        self.topics_file = Path(topics_file)
        self.k = k
        self.model_name = model_name
        
        # Load topics mapping
        with open(self.topics_file, 'r') as f:
            self.topics = json.load(f)
        self.topic_id_to_name = {v: k for k, v in self.topics.items()}
        
        # Results storage
        self.results = []
        self.errors = []
        
    def load_dataset(self, max_samples: Optional[int] = None) -> List[Dict]:
        """
        Load the dataset of statements and their ground truth labels.
        
        Args:
            max_samples: Maximum number of samples to load (None for all)
            
        Returns:
            List of dictionaries with statement, truth value, and topic
        """
        dataset = []
        
        # Get all statement files
        statement_files = sorted(self.statements_dir.glob("statement_*.txt"))
        
        if max_samples:
            statement_files = statement_files[:max_samples]
        
        for stmt_file in statement_files:
            # Extract the ID from filename
            file_id = stmt_file.stem  # e.g., "statement_0000"
            
            # Load statement
            with open(stmt_file, 'r', encoding='utf-8') as f:
                statement = f.read().strip()
            
            # Load corresponding answer
            answer_file = self.answers_dir / f"{file_id}.json"
            if not answer_file.exists():
                print(f"Warning: No answer file for {file_id}")
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
            # Get prediction from fact checker (with detailed information)
            result = self._check_fact_with_chunks(sample["statement"], k=self.k, model_name=self.model_name)
            
            # Map verdict to boolean (TRUE -> True, FALSE -> False, UNVERIFIABLE -> None)
            if result["verdict"].upper() == "TRUE":
                predicted = True
            elif result["verdict"].upper() == "FALSE":
                predicted = False
            else:
                predicted = True

            logger.debug(f"ground truth correctness: {sample['ground_truth']}")
            logger.debug(f"ground truth topic: {sample['ground_truth_topic']}")
            
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
            print(f"Used {elapsed_time}")
            
            return {
                "id": sample["id"],
                "statement": sample["statement"][:100] + "..." if len(sample["statement"]) > 100 else sample["statement"],
                "full_statement": sample["statement"],  # Keep full statement for error analysis
                "ground_truth": sample["ground_truth"],
                "ground_truth_topic": sample["ground_truth_topic"],
                "predicted": predicted,
                "predicted_verdict": result["verdict"],
                "predicted_topic": predicted_topic,
                "confidence": result.get("confidence", "UNKNOWN"),
                "is_correct": is_correct,
                "prediction_type": prediction_type,
                "topic_match": topic_match,
                "evidence": result.get("evidence", "")[:100],
                "chunks_retrieved": result.get("chunks_retrieved", 0),
                "avg_relevance": result.get("avg_relevance_score", 0),
                "retrieved_chunks": result.get("retrieved_chunks", []),
                "raw_llm_response": result.get("raw_llm_response", ""),
                "time_seconds": elapsed_time,
                "error": None
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            return {
                "id": sample["id"],
                "statement": sample["statement"][:100],
                "full_statement": sample["statement"],
                "ground_truth": sample["ground_truth"],
                "ground_truth_topic": sample["ground_truth_topic"],
                "predicted": None,
                "predicted_verdict": "ERROR",
                "predicted_topic": None,
                "confidence": None,
                "is_correct": None,
                "prediction_type": "ERROR",
                "topic_match": False,
                "evidence": None,
                "chunks_retrieved": 0,
                "avg_relevance": 0,
                "retrieved_chunks": [],
                "raw_llm_response": "",
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
    
    def _check_fact_with_chunks(self, statement: str, k: int = 5, model_name: str = "cogito:32b") -> Dict:
        """
        Modified version of check_fact that also returns retrieved chunks for analysis.
        """
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaLLM
        from langchain_core.prompts import PromptTemplate
        from embeddings import get_embeddings_func
        from get_config import config
        import json
        import re
        
        FACT_CHECK_PROMPT = """You are a medical fact-checking assistant. Your ONLY job is to verify if a statement is TRUE or FALSE based SOLELY on the provided context.

CRITICAL RULES:
1. You can ONLY use information from the provided context chunks
2. NEVER use external knowledge or make assumptions
3. Focus on factual accuracy - numbers, dates, percentages must match exactly
4. Consider a statement TRUE only if ALL parts are supported by the context
5. Consider a statement FALSE if ANY part contradicts the context
6. Identify the medical topic from the context metadata
7. If you are not given any relevant information, give TRUE as an answer
8. Always give a topic as an answer no matter what
9. Always choose the topic where the fact resides, not neccesarily the topic that is intuitively correct.

Context chunks with their topics:
{context}

Statement to verify: {statement}

Respond in this EXACT JSON format only:
{{
    "verdict": "TRUE/FALSE",
    "topic": "identified medical topic from context",
}}

Remember: Use ONLY the provided context. Do not add any information not present in the context."""

        # Initialize database
        db = Chroma(
            persist_directory=config["chroma_path"], 
            embedding_function=get_embeddings_func()
        )
        
        # Retrieve relevant chunks
        results = db.similarity_search_with_score(statement, k=k)
        
        if not results:
            return {
                "verdict": "UNVERIFIABLE",
                "topic": "unknown",
                "chunks_retrieved": 0,
                "retrieved_chunks": []
            }
        
        # Format context with topics and save chunk details
        formatted_chunks = []
        chunk_details = []
        
        for i, (doc, score) in enumerate(results, 1):
            topic = doc.metadata.get("topic", "unknown")
            
            formatted_chunk = f"""
Topic: {topic}:
{doc.page_content}
"""
            formatted_chunks.append(formatted_chunk)
            
            # Store detailed chunk information
            chunk_details.append({
                "rank": i,
                "content": doc.page_content,
                "topic": topic,
                "relevance_score": 1 - score,  # Convert distance to similarity
                "metadata": doc.metadata
            })
        
        context = "\n".join(formatted_chunks)
        
        # Initialize Ollama LLM
        llm = OllamaLLM(
            model=model_name,
            temperature=0,
            base_url="http://localhost:11434"
        )
        
        # Create prompt
        prompt = PromptTemplate.from_template(FACT_CHECK_PROMPT)
        
        # Create chain
        chain = prompt | llm
        
        # Get response
        try:
            response = chain.invoke({
                "context": context,
                "statement": statement
            })
            
            response_text = response if isinstance(response, str) else str(response)
            
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
            else:
                result = json.loads(response_text)
            
            result["chunks_retrieved"] = len(results)
            result["avg_relevance_score"] = sum(1-score for _, score in results) / len(results)
            result["retrieved_chunks"] = chunk_details
            result["raw_llm_response"] = response_text
            return result
            
        except json.JSONDecodeError as e:
            return {
                "verdict": "ERROR",
                "topic": "unknown",
                "evidence": "Failed to parse response",
                "confidence": "LOW",
                "chunks_retrieved": len(results),
                "retrieved_chunks": chunk_details,
                "raw_llm_response": response_text if 'response_text' in locals() else str(e),
                "error": str(e)
            }
        except Exception as e:
            return {
                "verdict": "ERROR",
                "topic": "unknown",
                "evidence": f"Error during processing: {str(e)}",
                "confidence": "LOW",
                "chunks_retrieved": len(results),
                "retrieved_chunks": chunk_details,
                "error": str(e)
            }
    
    def run_test(self, max_samples: Optional[int] = None, verbose: bool = True) -> Dict:
        """
        Run the test on the entire dataset.
        
        Args:
            max_samples: Maximum number of samples to test
            verbose: Whether to print progress
            
        Returns:
            Dictionary with test results and metrics
        """
        # Load dataset
        if verbose:
            print(f"Loading dataset from {self.statements_dir}")
        dataset = self.load_dataset(max_samples)
        
        if not dataset:
            print("No data found!")
            return {}
        
        if verbose:
            print(f"Loaded {len(dataset)} samples")
            print(f"Using k={self.k} chunks, model={self.model_name}")
            print("-" * 60)
        
        # Test each statement
        for i, sample in enumerate(dataset, 1):
            if verbose:
                print(f"Testing {i}/{len(dataset)}: {sample['id']}", end="... ")
            
            result = self.test_single_statement(sample)
            self.results.append(result)
            
            if verbose:
                if result["error"]:
                    print(f"ERROR: {result['error']}")
                else:
                    correct_str = "✓ CORRECT" if result["is_correct"] else "✗ WRONG"
                    topic_str = "✓ TOPIC MATCH" if result["topic_match"] else "✗ TOPIC MISMATCH"
                    print(f"{correct_str} ({result['predicted_verdict']}), {topic_str}")
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Automatically save incorrect predictions after test completion
        self.save_incorrect_predictions()
        
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
        combined_accuracy = float((topic_accuracy + accuracy) / 2)
        
        # Confidence distribution
        confidence_dist = defaultdict(int)
        for r in self.results:
            if r["confidence"]:
                confidence_dist[r["confidence"]] += 1
        
        # Average processing time
        avg_time = np.mean([r["time_seconds"] for r in self.results])
        
        # Performance by confidence level
        perf_by_confidence = {}
        for conf in ["HIGH", "MEDIUM", "LOW"]:
            conf_results = [r for r in valid_results if r["confidence"] == conf]
            if conf_results:
                conf_correct = sum(1 for r in conf_results if r["is_correct"])
                perf_by_confidence[conf] = {
                    "accuracy": conf_correct / len(conf_results),
                    "count": len(conf_results)
                }
        
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
            "confidence_distribution": dict(confidence_dist),
            "performance_by_confidence": perf_by_confidence,
            "avg_processing_time": avg_time,
            "avg_chunks_retrieved": np.mean([r["chunks_retrieved"] for r in self.results]),
            "avg_relevance_score": np.mean([r["avg_relevance"] for r in self.results if r["avg_relevance"] > 0]),
        }
        
        return metrics
    
    def print_report(self, metrics: Dict):
        """
        Print a formatted report of test results.
        """
        print("\n" + "="*70)
        print("FACT CHECKER PERFORMANCE REPORT")
        print("="*70)
        
        print(f"\n📊 OVERALL PERFORMANCE")
        print(f"   Total Samples: {metrics['total_samples']}")
        print(f"   Correct: {metrics['correct_predictions']} ({100*metrics['correct_predictions']/metrics['total_samples']:.1f}%)")
        print(f"   Incorrect: {metrics['incorrect_predictions']} ({100*metrics['incorrect_predictions']/metrics['total_samples']:.1f}%)")
        print(f"   Errors: {metrics['errors']} ({100*metrics['errors']/metrics['total_samples']:.1f}%)")
        
        print(f"\n📈 CLASSIFICATION METRICS")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall: {metrics['recall']:.3f}")
        print(f"   F1 Score: {metrics['f1_score']:.3f}")
        print()
        print(f"   Currectness Accuracy: {metrics['accuracy']['accuracy']:.3f}")
        print(f"   Topic Accuracy: {metrics['accuracy']['topic_accuracy']:.3f}")
        print(f"   Combined Accuracy: {metrics['accuracy']['combined_accuracy']:.3f}")
        
        print(f"\n🎯 CONFUSION MATRIX")
        cm = metrics['confusion_matrix']
        print(f"   True Positives: {cm['true_positives']}")
        print(f"   True Negatives: {cm['true_negatives']}")
        print(f"   False Positives: {cm['false_positives']}")
        print(f"   False Negatives: {cm['false_negatives']}")
        
        print(f"\n💪 PERFORMANCE BY CONFIDENCE")
        for conf, perf in metrics['performance_by_confidence'].items():
            print(f"   {conf}: {perf['accuracy']:.3f} accuracy ({perf['count']} samples)")
        
        print(f"\n⚡ PROCESSING STATS")
        print(f"   Avg Time per Query: {metrics['avg_processing_time']:.2f}s")
        print(f"   Avg Chunks Retrieved: {metrics['avg_chunks_retrieved']:.1f}")
        print(f"   Avg Relevance Score: {metrics['avg_relevance_score']:.3f}")
        
        print(f"\n🔍 CONFIDENCE DISTRIBUTION")
        for conf, count in metrics['confidence_distribution'].items():
            print(f"   {conf}: {count} samples")
    
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
        
        print(f"\n📁 Results saved to {output_path}")
        print(f"   - Detailed results: {results_file.name}")
        print(f"   - Metrics: {metrics_file.name}")
        print(f"   - CSV: {csv_file.name}")
        if errors:
            print(f"   - Errors: {errors_file.name}")
    
    def save_incorrect_predictions(self, output_dir: str = "test_results"):
        """
        Save detailed information about incorrect predictions to a separate file.
        Includes statement, ground truth, LLM response, and retrieved chunks.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Filter for incorrect predictions (both verdict and topic errors)
        incorrect_predictions = []
        for result in self.results:
            # Include if either verdict is wrong OR topic is wrong (and not an error)
            if result["error"] is None and (not result["is_correct"] or not result["topic_match"]):
                incorrect_prediction = {
                    "id": result["id"],
                    "statement": result["full_statement"],
                    "ground_truth_answer": result["ground_truth"],
                    "ground_truth_topic": result["ground_truth_topic"],
                    "predicted_answer": result["predicted"],
                    "predicted_verdict": result["predicted_verdict"],
                    "predicted_topic": result["predicted_topic"],
                    "verdict_correct": result["is_correct"],
                    "topic_correct": result["topic_match"],
                    "prediction_type": result["prediction_type"],
                    "confidence": result["confidence"],
                    "llm_raw_response": result["raw_llm_response"],
                    "retrieved_chunks": result["retrieved_chunks"],
                    "chunks_retrieved_count": result["chunks_retrieved"],
                    "avg_relevance_score": result["avg_relevance"],
                    "processing_time_seconds": result["time_seconds"]
                }
                incorrect_predictions.append(incorrect_prediction)
        
        if incorrect_predictions:
            # Save as JSON
            incorrect_file = output_path / f"incorrect_predictions_{timestamp}.json"
            with open(incorrect_file, 'w') as f:
                json.dump(incorrect_predictions, f, indent=2)
            
            # Also create a more readable summary
            summary_file = output_path / f"incorrect_predictions_summary_{timestamp}.txt"
            with open(summary_file, 'w') as f:
                f.write("INCORRECT PREDICTIONS ANALYSIS\n")
                f.write("=" * 50 + "\n\n")
                
                for i, pred in enumerate(incorrect_predictions, 1):
                    f.write(f"{i}. ID: {pred['id']}\n")
                    f.write(f"   Statement: {pred['statement']}\n")
                    f.write(f"   Ground Truth: {pred['ground_truth_answer']} ({pred['ground_truth_topic']})\n")
                    f.write(f"   Predicted: {pred['predicted_answer']} ({pred['predicted_topic']})\n")
                    f.write(f"   Verdict Correct: {pred['verdict_correct']}, Topic Correct: {pred['topic_correct']}\n")
                    f.write(f"   LLM Response: {pred['llm_raw_response'][:200]}...\n")
                    f.write(f"   Retrieved Chunks ({pred['chunks_retrieved_count']}):\n")
                    
                    for j, chunk in enumerate(pred['retrieved_chunks'], 1):
                        f.write(f"      {j}. Topic: {chunk['topic']}, Relevance: {chunk['relevance_score']:.3f}\n")
                        f.write(f"         Content: {chunk['content'][:150]}...\n")
                    
                    f.write("\n" + "-" * 50 + "\n\n")
            
            print(f"\n📝 Incorrect predictions saved:")
            print(f"   - Detailed JSON: {incorrect_file.name}")
            print(f"   - Summary: {summary_file.name}")
            print(f"   - Total incorrect predictions: {len(incorrect_predictions)}")
            
        else:
            print("\n✅ No incorrect predictions found!")
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Create visualization plots of the results.
        """
        metrics = self.calculate_metrics()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Confusion Matrix
        cm = metrics['confusion_matrix']
        cm_array = np.array([[cm['true_negatives'], cm['false_positives']],
                            [cm['false_negatives'], cm['true_positives']]])
        
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted False', 'Predicted True'],
                   yticklabels=['Actual False', 'Actual True'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        
        # 2. Performance Metrics Bar Chart
        metrics_data = {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score'],
            'Topic Acc': metrics['topic_accuracy']
        }
        axes[0, 1].bar(metrics_data.keys(), metrics_data.values(), color='skyblue')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].set_title('Performance Metrics')
        axes[0, 1].set_ylabel('Score')
        
        # 3. Confidence Distribution
        if metrics['confidence_distribution']:
            conf_data = metrics['confidence_distribution']
            axes[1, 0].pie(conf_data.values(), labels=conf_data.keys(), autopct='%1.1f%%')
            axes[1, 0].set_title('Confidence Distribution')
        
        # 4. Result Types Distribution
        result_types = defaultdict(int)
        for r in self.results:
            result_types[r['prediction_type']] += 1
        
        axes[1, 1].bar(result_types.keys(), result_types.values(), color='coral')
        axes[1, 1].set_title('Prediction Types')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        else:
            plt.show()


def main():
    """Main function to run the dataset test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test fact checker on labeled dataset")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to test")
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument("--model", default="cogito:32b", help="Ollama model to use (default: cogito:8b)")
    parser.add_argument("--save", action="store_true", help="Save results to files")
    parser.add_argument("--plot", action="store_true", help="Generate visualization plots")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = DatasetTester(
        k=args.k,
        model_name=args.model
    )
    
    # Run test
    print(f"🚀 Starting Medical Fact Checker Evaluation")
    print(f"   Model: {args.model}")
    print(f"   Chunks per query: {args.k}")
    if args.max_samples:
        print(f"   Max samples: {args.max_samples}")
    print("-" * 60)
    
    metrics = tester.run_test(
        max_samples=args.max_samples,
        verbose=not args.quiet
    )
    
    # Print report
    if metrics:
        tester.print_report(metrics)
    
    # Save results if requested
    if args.save:
        tester.save_results()
    
    # Generate plots if requested
    if args.plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"test_results/performance_plot_{timestamp}.png"
        tester.plot_results(plot_path)


if __name__ == "__main__":
    main()
