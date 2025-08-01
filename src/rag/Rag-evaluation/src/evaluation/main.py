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
from ragas.llms.base import llm_factory
from ragas.embeddings.base import embedding_factory

# Add project root to Python path to resolve relative imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.templates.simple_rag import SimpleRAG
from src.templates.hybrid_rag import HybridRAG
from src.templates.hyde import HyDE
from src.templates.contextual_retriever import ContextualRetrieverRAG
# from src.templates.graph_rag_graph_retriever import GraphRAG
# from src.templates.graph_rag_hybrid_retriever import GraphHybridRAG
from src.templates.query_expansion_with_rrf import QueryExpansionRRF
from src.templates.query_rewrite_rag import QueryRewriteRAG
from src.templates.semantic_chunker import SemanticChunkerRAG
from src.templates.step_back_prompt import StepBackPromptRAG
from src.evaluation.config import EVALUATE_METHODS, RUN_TAG, RAGAS_METRICS
from src.evaluation.metrics import (
    calculate_context_overlap,
    calculate_precision_at_k,
    calculate_recall_at_k,
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

# Environment sanity checks / fallbacks
os.environ.setdefault("USER_AGENT", "rag-holygrail-evaluator/0.1")
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = "http://localhost:1984"
os.environ["LANGCHAIN_API_KEY"] = "YOUR_API_KEY"
os.environ["LANGCHAIN_HUB_API_URL"] = "http://localhost:1984"

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env file")

def run_template_worker(template_name: str,
                        template_cls,
                        questions_df: pd.DataFrame) -> tuple[str, list[dict]]:
    """
    Runs one template over all questions inside a separate process.
    Returns (template_name, results_list)
    """
    llm = llm_factory()
    embeddings = embedding_factory()
    template = template_cls(llm=llm, embeddings=embeddings)

    results = []
    for _, row in questions_df.iterrows():
        start_time = time.time()
        try:
            result = template.run(row['question'], row['reference_contexts'])
            answer = result['answer']
            retrieved_context = result['context']
            if isinstance(retrieved_context, str):
                retrieved_context = [retrieved_context]
            response_time = time.time() - start_time
        except Exception as exc:
            answer = "Error occurred during processing"
            retrieved_context = []
            response_time = 0.0
            print(f"[{template_name}] Exception: {exc}")
            traceback.print_exc()

        results.append({
            "question": row["question"],
            "answer": answer,
            "retrieved_context": retrieved_context,
            "reference_contexts": row["reference_contexts"],
            "response_time": response_time,
            "ground_truths": row["ground_truths"],
        })

    return template_name, results

def run_ragas_worker(template_name: str,
                     results: list[dict],
                     ragas_metrics):
    """
    Compute RAGAS metrics for a single template in a separate process.
    Returns (template_name, {metric_name: score, ...})
    """
    from datasets import Dataset
    from ragas import evaluate

    eval_data = {
        "question":     [r["question"] for r in results],
        "answer":       [r["answer"] for r in results],
        "contexts":     [r["retrieved_context"] for r in results],
        "ground_truth": [r["ground_truths"][0] for r in results],
    }
    ds = Dataset.from_dict(eval_data)

    try:
        ragas_result = evaluate(ds, metrics=ragas_metrics)
        df_res = ragas_result.to_pandas()
        scores = {m.name: float(df_res[m.name].mean()) for m in ragas_metrics}
    except Exception as exc:
        print(f"[RAGAS‑{template_name}] Evaluation failed: {exc}")
        scores = {m.name: 0.0 for m in ragas_metrics}

    return template_name, scores

def main():
    print("Loading testset...")
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    testset_path = os.path.join(project_root, 'data', 'datasets', 'testset.json')
    components_path = os.path.join(project_root, 'data', 'components.json')

    with open(testset_path, 'r') as f:
        testset_data = json.load(f)
    with open(components_path, 'r') as f:
        components_data = json.load(f)
    
    NUM_QUESTIONS = len(testset_data)
    NUM_COMPONENTS = len(components_data)

    dataset = [{'question': item['question'], 'ground_truths': [item['ground_truth']], 'reference_contexts': item['contexts']} for item in testset_data]
    df = pd.DataFrame(dataset)
    print(f"Loaded {len(df)} test cases")

    print("Initializing RAG templates...")

    all_templates = {
        "simple_rag": SimpleRAG,
        "hybrid_rag": HybridRAG,
        "hyde": HyDE,
        "contextual_retriever": ContextualRetrieverRAG,
        "query_expansion_with_rrf": QueryExpansionRRF,
        "query_rewrite_rag": QueryRewriteRAG,
        "query_rewrite": QueryRewriteRAG,
        "rrf": QueryExpansionRRF,
        "semantic_chunker": SemanticChunkerRAG,
        "step_back_prompt": StepBackPromptRAG,
        "step_back_prompting": StepBackPromptRAG,
    }
    templates_to_run = {name: all_templates[name] for name in EVALUATE_METHODS if name in all_templates}

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

    with ProcessPoolExecutor(max_workers=min(cpu_count(), len(template_results))) as pool:
        ragas_futures = [
            pool.submit(run_ragas_worker, name, results, RAGAS_METRICS)
            for name, results in template_results.items()
        ]

        done, _ = wait(ragas_futures, return_when=ALL_COMPLETED)
        for fut in done:
            try:
                name, scores = fut.result()
                template_ragas_scores[name] = scores
                print(f"  {name} RAGAS:", ", ".join(f"{k}={v:.3f}" for k, v in scores.items()))
            except Exception as exc:
                print(f"[main] RAGAS worker failed: {exc}")

    print("Calculating additional metrics...")
    template_additional_scores = {}
    for template_name, results in template_results.items():
        print(f"Calculating additional metrics for {template_name}...")
        
        context_overlaps, precision_at_k_scores, recall_at_k_scores, f1_scores = [], [], [], []
        response_times = [item['response_time'] for item in results]
        
        for item in results:
            overlap = calculate_context_overlap(item['reference_contexts'], item['retrieved_context'])
            context_overlaps.append(overlap)
            
            precision_at_k = calculate_precision_at_k(item['reference_contexts'], item['retrieved_context'], 5)
            precision_at_k_scores.append(precision_at_k)
            
            recall_at_k = calculate_recall_at_k(item['reference_contexts'], item['retrieved_context'], 5)
            recall_at_k_scores.append(recall_at_k)
            
            f1 = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k) if (precision_at_k + recall_at_k) > 0 else 0
            f1_scores.append(f1)
        
        template_additional_scores[template_name] = {
            'context_overlap': np.mean(context_overlaps),
            'precision_at_k': np.mean(precision_at_k_scores),
            'recall_at_k': np.mean(recall_at_k_scores),
            'f1_score': np.mean(f1_scores),
            'response_time': np.mean(response_times)
        }
        
        print(f"  {template_name} additional scores:")
        for metric, score in template_additional_scores[template_name].items():
            print(f"    {metric}: {score}")

    print("\nCreating ranking table...")
    ranking_table = create_ranking_table(template_ragas_scores, template_additional_scores)
    print("\nRanking Table:")
    print(ranking_table.to_string(index=False))

    ranked_templates = get_ranked_templates(template_ragas_scores, template_additional_scores)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_methods = len(EVALUATE_METHODS)
    run_folder_name = f"M{num_methods}_Q{NUM_QUESTIONS}_C{NUM_COMPONENTS}_{RUN_TAG}_{timestamp}"
    output_dir = os.path.join("results", run_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    create_radar_chart(template_ragas_scores, output_dir, RAGAS_METRICS)
    create_bar_charts(template_additional_scores, output_dir)

    # Perform a statistical test only if at least two templates are available
    if len(template_results) >= 2:
        # Try the canonical pair first, otherwise pick the first two arbitrary keys
        if "simple_rag" in template_results and "hybrid_rag" in template_results:
            perform_statistical_test(template_results["simple_rag"], template_results["hybrid_rag"])
        else:
            first, second = list(template_results.keys())[:2]
            perform_statistical_test(template_results[first], template_results[second])
    else:
        print("Not enough templates to run a statistical significance test — skipping.")

    provide_recommendations(ranked_templates, template_ragas_scores, template_additional_scores, RAGAS_METRICS)
    save_results(template_results, template_ragas_scores, template_additional_scores, ranked_templates, output_dir)

    summary_data = aggregate_results()
    create_comparison_charts(summary_data)

    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()
