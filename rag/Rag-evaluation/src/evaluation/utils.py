import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


def create_ranking_table(template_ragas_scores, template_additional_scores):
    data = []
    for template_name in template_ragas_scores.keys():
        ragas_scores = template_ragas_scores[template_name]
        additional_scores = template_additional_scores[template_name]
        data.append(
            {
                "Template": template_name,
                "Context Recall": ragas_scores.get("context_recall", 0.0),
                "Context Precision": ragas_scores.get("context_precision", 0.0),
                "Faithfulness": ragas_scores.get("faithfulness", 0.0),
                "Answer Relevancy": ragas_scores.get("answer_relevancy", 0.0),
                "Context Overlap": additional_scores["context_overlap"],
                "Precision@K": additional_scores["precision_at_k"],
                "Recall@K": additional_scores["recall_at_k"],
                "F1 Score": additional_scores["f1_score"],
                "Response Time": additional_scores["response_time"],
            }
        )
    return pd.DataFrame(data)


def create_radar_chart(template_ragas_scores, output_dir, ragas_metrics):
    categories = [metric.name for metric in ragas_metrics]

    fig = go.Figure()

    for template_name in template_ragas_scores.keys():
        scores = template_ragas_scores[template_name]
        fig.add_trace(
            go.Scatterpolar(
                r=[scores.get(k, 0.0) for k in categories],
                theta=categories,
                fill="toself",
                name=template_name.replace("_", " ").title(),
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="RAGAS Metrics Comparison",
    )

    fig.write_html(os.path.join(output_dir, "ragas_metrics_radar_chart.html"))
    print(
        f"Radar chart saved to {os.path.join(output_dir, 'ragas_metrics_radar_chart.html')}"
    )


def create_bar_charts(template_additional_scores, output_dir):
    metrics = list(
        template_additional_scores[list(template_additional_scores.keys())[0]].keys()
    )
    template_names = list(template_additional_scores.keys())

    fig = make_subplots(rows=len(metrics), cols=1, subplot_titles=metrics)

    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Bar(
                x=template_names,
                y=[
                    template_additional_scores[template_name][metric]
                    for template_name in template_names
                ],
                name=metric,
                showlegend=False,
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(height=1200, title_text="Additional Metrics Comparison")

    fig.write_html(os.path.join(output_dir, "additional_metrics_bar_charts.html"))
    print(
        f"Bar charts saved to {os.path.join(output_dir, 'additional_metrics_bar_charts.html')}"
    )


def perform_statistical_test(template1_results, template2_results):
    template1_response_times = [item["response_time"] for item in template1_results]
    template2_response_times = [item["response_time"] for item in template2_results]

    t_statistic, p_value = stats.ttest_ind(
        template1_response_times, template2_response_times
    )

    print("\nStatistical Test (Simple RAG vs Hybrid RAG):")
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")

    if p_value < 0.05:
        print("The difference in response times is statistically significant.")
    else:
        print("The difference in response times is not statistically significant.")


def get_ranked_templates(
    template_ragas_scores,
    template_additional_scores,
    recall_weight=0.3,
    precision_weight=0.3,
    faithfulness_weight=0.4,
):
    template_scores = {}
    for template_name in template_ragas_scores.keys():
        ragas_scores = template_ragas_scores[template_name]

        weighted_score = (
            recall_weight * ragas_scores.get("context_recall", 0.0)
            + precision_weight * ragas_scores.get("context_precision", 0.0)
            + faithfulness_weight * ragas_scores.get("faithfulness", 0.0)
        )

        template_scores[template_name] = weighted_score

    ranked_templates = sorted(template_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_templates


def provide_recommendations(
    ranked_templates, template_ragas_scores, template_additional_scores, ragas_metrics
):
    best_template = ranked_templates[0][0]
    print(f"\nRecommended Template: {best_template}\n")

    print("Strengths Analysis:")
    ragas_scores = template_ragas_scores[best_template]
    additional_scores = template_additional_scores[best_template]
    for metric in ragas_metrics:
        print(f"- {metric.name}: {ragas_scores.get(metric.name, 0.0):.4f}")
    for metric, score in additional_scores.items():
        print(f"- {metric}: {score:.4f}")

    print("\nProduction Deployment Recommendations:")
    print("- Consider the trade-offs between recall, precision, and response time.")
    print(
        "- Monitor the performance of the RAG template in production and adjust the weights accordingly."
    )
    print(
        "- Continuously evaluate the RAG template with new test cases to ensure its effectiveness."
    )


def save_results(
    template_results,
    template_ragas_scores,
    template_additional_scores,
    ranked_templates,
    output_dir,
):
    results = {
        "template_results": {
            k: [
                {**i, "retrieved_context": [str(c) for c in i["retrieved_context"]]}
                for i in v
            ]
            for k, v in template_results.items()
        },
        "template_ragas_scores": template_ragas_scores,
        "template_additional_scores": template_additional_scores,
        "ranked_templates": ranked_templates,
    }

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "rag_evaluation_results.json")

    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {filename}")


def aggregate_results(results_dir="results"):
    summary_data = []
    for run_folder in os.listdir(results_dir):
        run_path = os.path.join(results_dir, run_folder)
        if os.path.isdir(run_path):
            results_file = os.path.join(run_path, "rag_evaluation_results.json")
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    data = json.load(f)

                    for template_name, scores in data["template_ragas_scores"].items():
                        additional_scores = data["template_additional_scores"][
                            template_name
                        ]
                        summary_data.append(
                            {
                                "run_folder": run_folder,
                                "template": template_name,
                                "context_recall": scores.get("context_recall"),
                                "context_precision": scores.get("context_precision"),
                                "answer_relevancy": scores.get("answer_relevancy"),
                                "response_time": additional_scores.get("response_time"),
                                "avg_score": np.mean(
                                    [
                                        scores.get("context_recall", 0),
                                        scores.get("context_precision", 0),
                                        scores.get("answer_relevancy", 0),
                                    ]
                                ),
                            }
                        )

    summary_filename = os.path.join(results_dir, "evaluation_summary.json")
    with open(summary_filename, "w") as f:
        json.dump(summary_data, f, indent=4)
    print(f"Aggregated summary saved to {summary_filename}")
    return summary_data


def create_comparison_charts(summary_data, results_dir="results"):
    if not summary_data:
        print("No summary data to create comparison charts.")
        return

    df = pd.DataFrame(summary_data)

    metrics_to_plot = [
        "context_recall",
        "context_precision",
        "response_time",
        "answer_relevancy",
        "avg_score",
    ]

    fig = make_subplots(
        rows=len(metrics_to_plot), cols=1, subplot_titles=metrics_to_plot
    )

    for i, metric in enumerate(metrics_to_plot):
        for template_name in df["template"].unique():
            template_df = df[df["template"] == template_name]
            fig.add_trace(
                go.Bar(
                    x=template_df["run_folder"],
                    y=template_df[metric],
                    name=template_name,
                ),
                row=i + 1,
                col=1,
            )

    fig.update_layout(height=1500, title_text="Cross-Evaluation Comparison")
    comparison_filename = os.path.join(results_dir, "cross_evaluation_comparison.html")
    fig.write_html(comparison_filename)
    print(f"Comparison chart saved to {comparison_filename}")
