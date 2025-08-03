import os
import json
import time
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from multiprocessing import cpu_count
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate

# Add project root to Python path to resolve relative imports
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.templates.healthcare_rag import HealthcareRAG
from src.llm_client import LocalLLMClient

# from src.templates.pocketflow_rag import PocketFlowRAG
# from src.templates.simple_rag import SimpleRAG
# from src.templates.hybrid_rag import HybridRAG
# from src.templates.hyde import HyDE
# from src.templates.contextual_retriever import ContextualRetrieverRAG
# from src.templates.graph_rag_graph_retriever import GraphRAG
# from src.templates.graph_rag_hybrid_retriever import GraphHybridRAG
# from src.templates.query_expansion_with_rrf import QueryExpansionRRF
# from src.templates.query_rewrite_rag import QueryRewriteRAG
# from src.templates.semantic_chunker import SemanticChunkerRAG
# from src.templates.step_back_prompt import StepBackPromptRAG
from src.evaluation.config import EVALUATE_METHODS, RUN_TAG, RAGAS_METRICS
from src.evaluation.metrics import (
    calculate_context_overlap,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_binary_accuracy,
    calculate_topic_accuracy,
    calculate_overall_accuracy,
)
from src.evaluation.utils import (
    create_ranking_table,
    create_radar_chart,
    create_bar_charts,
    perform_statistical_test,
    save_results,
    aggregate_results,
    create_comparison_charts,
    provide_recommendations,
    get_ranked_templates,
)

# Load environment variables
load_dotenv()

# Using Ollama/Qwen 8B locally - no cloud services needed


def parse_ground_truth(ground_truth_str: str) -> tuple[int, int]:
    """
    Parse ground truth string like "Statement is true, topic: 4" into (statement_is_true, topic)

    Args:
        ground_truth_str: String like "Statement is true, topic: 4" or "Statement is false, topic: 63"

    Returns:
        Tuple of (statement_is_true, statement_topic)
    """
    try:
        # Extract true/false
        if "is true" in ground_truth_str.lower():
            statement_is_true = 1
        elif "is false" in ground_truth_str.lower():
            statement_is_true = 0
        else:
            statement_is_true = 1  # Default to true if unclear

        # Extract topic number
        import re

        topic_match = re.search(r"topic:\s*(\d+)", ground_truth_str.lower())
        if topic_match:
            statement_topic = int(topic_match.group(1))
        else:
            statement_topic = 0  # Default to topic 0 if unclear

        return statement_is_true, statement_topic
    except Exception:
        return 1, 0  # Safe defaults


def parse_answer(answer_str: str) -> tuple[int, int]:
    """
    Parse answer JSON string to extract statement_is_true and statement_topic

    Args:
        answer_str: JSON string like '{"statement_is_true": 1, "statement_topic": 4}'

    Returns:
        Tuple of (statement_is_true, statement_topic)
    """
    try:
        answer_dict = json.loads(answer_str)
        statement_is_true = int(answer_dict.get("statement_is_true", 1))
        statement_topic = int(answer_dict.get("statement_topic", 0))

        # Ensure values are in valid ranges
        statement_is_true = max(0, min(1, statement_is_true))
        statement_topic = max(0, min(114, statement_topic))

        return statement_is_true, statement_topic
    except Exception:
        return 1, 0  # Safe defaults


def calculate_custom_context_recall(results: list[dict]) -> float:
    """
    Calculate a custom context recall metric appropriate for classification tasks.
    This measures how well the retrieved contexts relate to the question content,
    rather than trying to find ground truth labels in the contexts.
    """
    total_recall = 0.0
    for item in results:
        question = item["question"]
        retrieved_contexts = item["retrieved_context"]

        if not retrieved_contexts or not question:
            continue

        # Extract key terms from the question
        question_words = set(question.lower().split())
        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
        }
        question_words = question_words - stop_words

        if not question_words:
            continue

        # Check how many question terms appear in retrieved contexts
        context_words = set()
        for ctx in retrieved_contexts:
            context_words.update(ctx.lower().split())

        # Calculate recall as overlap between question terms and context terms
        overlap = len(question_words.intersection(context_words))
        recall = overlap / len(question_words) if question_words else 0.0
        total_recall += recall

    return total_recall / len(results) if results else 0.0


def run_template_worker(
    template_name: str, template_cls, questions_df: pd.DataFrame
) -> tuple[str, list[dict]]:
    """
    Runs one template over all questions inside a separate process.
    Returns (template_name, results_list)
    """
    llm_client = LocalLLMClient(model_name="llama3.2:latest")
    llm_client.ensure_model_available()
    template = template_cls(llm_client=llm_client)

    results = []
    for _, row in questions_df.iterrows():
        start_time = time.time()
        try:
            result = template.run(row["question"], row["reference_contexts"])
            answer = result["answer"]
            retrieved_context = result["context"]
            if isinstance(retrieved_context, str):
                retrieved_context = [retrieved_context]
            response_time = time.time() - start_time
        except Exception as exc:
            answer = "Error occurred during processing"
            retrieved_context = []
            response_time = 0.0
            print(f"[{template_name}] Exception: {exc}")
            traceback.print_exc()

        results.append(
            {
                "question": row["question"],
                "answer": answer,
                "retrieved_context": retrieved_context,
                "reference_contexts": row["reference_contexts"],
                "response_time": response_time,
                "ground_truths": row["ground_truths"],
                "parsed_answer": None,  # Will be filled later
            }
        )

    return template_name, results


def run_ragas_worker(template_name: str, results: list[dict], ragas_metrics):
    """
    Compute RAGAS metrics for a single template in a separate process.
    Returns (template_name, {metric_name: score, ...})
    """

    eval_data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["retrieved_context"] for r in results],
        "ground_truth": [r["ground_truths"][0] for r in results],
    }
    ds = Dataset.from_dict(eval_data)

    try:
        # Skip context_recall for classification tasks as it's not applicable
        # The ground truth is a classification label, not content to be found in contexts
        classification_metrics = [
            m for m in ragas_metrics if m.name != "context_recall"
        ]
        if classification_metrics:
            ragas_result = evaluate(ds, metrics=classification_metrics)
            df_res = ragas_result.to_pandas()
            scores = {
                m.name: float(df_res[m.name].mean()) for m in classification_metrics
            }
            # Set context_recall to a reasonable value based on context overlap
            scores["context_recall"] = calculate_custom_context_recall(results)
        else:
            scores = {m.name: 0.0 for m in ragas_metrics}
    except Exception as exc:
        print(f"[RAGAS‑{template_name}] Evaluation failed: {exc}")
        scores = {m.name: 0.0 for m in ragas_metrics}

    return template_name, scores


def main():
    print("Loading testset...")
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    testset_path = os.path.join(project_root, "data", "datasets", "testset.json")
    components_path = os.path.join(project_root, "data", "components.json")

    with open(testset_path, "r") as f:
        testset_data = json.load(f)
    with open(components_path, "r") as f:
        components_data = json.load(f)

    NUM_QUESTIONS = len(testset_data)
    NUM_COMPONENTS = len(components_data)

    dataset = [
        {
            "question": item["question"],
            "ground_truths": [item["ground_truth"]],
            "reference_contexts": item["contexts"],
        }
        for item in testset_data
    ]
    df = pd.DataFrame(dataset[:10])  # Limit to first 10 test cases
    print(f"Loaded {len(df)} test cases")

    print("Initializing RAG templates...")

    all_templates = {
        "healthcare_rag": HealthcareRAG,
        # "pocketflow_rag": PocketFlowRAG,
        # "simple_rag": SimpleRAG,
        # "hybrid_rag": HybridRAG,
        # "hyde": HyDE,
        # "contextual_retriever": ContextualRetrieverRAG,
        # "query_expansion_with_rrf": QueryExpansionRRF,
        # "query_rewrite_rag": QueryRewriteRAG,
        # "query_rewrite": QueryRewriteRAG,
        # "rrf": QueryExpansionRRF,
        # "semantic_chunker": SemanticChunkerRAG,
        # "step_back_prompt": StepBackPromptRAG,
        # "step_back_prompting": StepBackPromptRAG,
    }
    templates_to_run = {
        name: all_templates[name] for name in EVALUATE_METHODS if name in all_templates
    }

    print("Templates requested:", EVALUATE_METHODS)
    print("Will run:", list(templates_to_run))

    print("Running RAG templates in parallel...")
    template_results = {}

    max_workers = min(cpu_count(), len(templates_to_run))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for name, cls in templates_to_run.items():
            futures.append(pool.submit(run_template_worker, name, cls, df))

        done, _ = wait(futures, return_when=ALL_COMPLETED)
        for fut in done:
            try:
                tpl_name, results = fut.result()
                template_results[tpl_name] = results
            except Exception as exc:
                print(f"[main] Worker failed: {exc}")

    print("Calculating RAGAS metrics in parallel...")
    template_ragas_scores = {}

    with ProcessPoolExecutor(
        max_workers=min(cpu_count(), len(template_results))
    ) as pool:
        ragas_futures = [
            pool.submit(run_ragas_worker, name, results, RAGAS_METRICS)
            for name, results in template_results.items()
        ]

        done, _ = wait(ragas_futures, return_when=ALL_COMPLETED)
        for fut in done:
            try:
                name, scores = fut.result()
                template_ragas_scores[name] = scores
                print(
                    f"  {name} RAGAS:",
                    ", ".join(f"{k}={v:.3f}" for k, v in scores.items()),
                )
            except Exception as exc:
                print(f"[main] RAGAS worker failed: {exc}")

    print("Calculating additional metrics...")
    template_additional_scores = {}
    for template_name, results in template_results.items():
        print(f"Calculating additional metrics for {template_name}...")

        context_overlaps, precision_at_k_scores, recall_at_k_scores, f1_scores = (
            [],
            [],
            [],
            [],
        )
        response_times = [item["response_time"] for item in results]

        for item in results:
            overlap = calculate_context_overlap(
                item["reference_contexts"], item["retrieved_context"]
            )
            context_overlaps.append(overlap)

            precision_at_k = calculate_precision_at_k(
                item["reference_contexts"], item["retrieved_context"], 5
            )
            precision_at_k_scores.append(precision_at_k)

            recall_at_k = calculate_recall_at_k(
                item["reference_contexts"], item["retrieved_context"], 5
            )
            recall_at_k_scores.append(recall_at_k)

            f1 = (
                2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)
                if (precision_at_k + recall_at_k) > 0
                else 0
            )
            f1_scores.append(f1)

        # Calculate accuracy metrics
        binary_predictions = []
        topic_predictions = []
        binary_ground_truth = []
        topic_ground_truth = []

        for item in results:
            # Parse ground truth
            gt_binary, gt_topic = parse_ground_truth(item["ground_truths"][0])
            binary_ground_truth.append(gt_binary)
            topic_ground_truth.append(gt_topic)

            # Parse prediction
            pred_binary, pred_topic = parse_answer(item["answer"])
            binary_predictions.append(pred_binary)
            topic_predictions.append(pred_topic)

        # Calculate accuracy scores
        binary_accuracy = calculate_binary_accuracy(
            binary_predictions, binary_ground_truth
        )
        topic_accuracy = calculate_topic_accuracy(topic_predictions, topic_ground_truth)
        overall_accuracy = calculate_overall_accuracy(
            binary_predictions,
            topic_predictions,
            binary_ground_truth,
            topic_ground_truth,
        )

        template_additional_scores[template_name] = {
            "context_overlap": np.mean(context_overlaps),
            "precision_at_k": np.mean(precision_at_k_scores),
            "recall_at_k": np.mean(recall_at_k_scores),
            "f1_score": np.mean(f1_scores),
            "response_time": np.mean(response_times),
            "binary_accuracy": binary_accuracy,
            "topic_accuracy": topic_accuracy,
            "overall_accuracy": overall_accuracy,
        }

        print(f"  {template_name} additional scores:")
        for metric, score in template_additional_scores[template_name].items():
            print(f"    {metric}: {score}")

    print("\nCreating ranking table...")
    ranking_table = create_ranking_table(
        template_ragas_scores, template_additional_scores
    )
    print("\nRanking Table:")
    print(ranking_table.to_string(index=False))

    ranked_templates = get_ranked_templates(
        template_ragas_scores, template_additional_scores
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_methods = len(EVALUATE_METHODS)
    run_folder_name = (
        f"M{num_methods}_Q{NUM_QUESTIONS}_C{NUM_COMPONENTS}_{RUN_TAG}_{timestamp}"
    )
    output_dir = os.path.join("results", run_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    create_radar_chart(template_ragas_scores, output_dir, RAGAS_METRICS)
    create_bar_charts(template_additional_scores, output_dir)

    # Perform a statistical test only if at least two templates are available
    if len(template_results) >= 2:
        # Try the canonical pair first, otherwise pick the first two arbitrary keys
        if "simple_rag" in template_results and "hybrid_rag" in template_results:
            perform_statistical_test(
                template_results["simple_rag"], template_results["hybrid_rag"]
            )
        else:
            first, second = list(template_results.keys())[:2]
            perform_statistical_test(template_results[first], template_results[second])
    else:
        print("Not enough templates to run a statistical significance test — skipping.")

    provide_recommendations(
        ranked_templates,
        template_ragas_scores,
        template_additional_scores,
        RAGAS_METRICS,
    )
    save_results(
        template_results,
        template_ragas_scores,
        template_additional_scores,
        ranked_templates,
        output_dir,
    )

    summary_data = aggregate_results()
    create_comparison_charts(summary_data)

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
