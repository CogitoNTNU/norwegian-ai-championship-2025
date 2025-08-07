from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from loguru import logger

from embeddings import get_embeddings_func
from get_config import config


FACT_CHECK_PROMPT = """You are a medical fact-checking assistant. Your ONLY job is to verify if a statement is TRUE, FALSE, or UNSURE based SOLELY on the provided context.

CRITICAL RULES:
1. You can ONLY use information from the provided context chunks
2. NEVER use external knowledge or make assumptions
3. Focus on factual accuracy - numbers, dates, percentages must match exactly
4. Consider a statement TRUE if ALL parts are supported by the context
5. Consider a statement FALSE if ANY part contradicts the context
6. Consider a statement UNSURE if the context is insufficient to determine truth or falsehood
7. For the topic field: You MUST identify the topic from the SPECIFIC context chunk that contains the evidence you used to make your verdict. Look at the "[Topic: X]" label in the context chunk where you found the supporting or contradicting information. Do NOT guess or choose what seems most intuitive - use the exact topic label from the evidence source.
8. Always give a topic as an answer no matter what
9. Choose the topic from the chunk which falsifies or confirms the statement

Context chunks with their topics:
{context}

Statement to verify: {statement}

Respond in this EXACT JSON format only:
{{
    "verdict": "TRUE/FALSE/UNSURE",
    "topic": "exact topic name from the context chunk that provided the evidence",
}}

Remember: Use ONLY the provided context. The topic must come from the specific context chunk that contains your evidence, not from general medical knowledge or intuition."""


def format_context_with_topics(results: List[Tuple]) -> str:
    """
    Format the search results to include topic information prominently.
    """
    formatted_chunks = []
    for i, (doc, score) in enumerate(results, 1):
        topic = doc.metadata.get("topic", "unknown")

        formatted_chunk = f"""
[Topic: {topic}]:
{doc.page_content}
---"""
        formatted_chunks.append(formatted_chunk)

    return "\n".join(formatted_chunks)


def check_fact(statement: str, model_name: str = None) -> Dict:
    """
    Check if a medical statement is true or false based on the knowledge base.

    Args:
        statement: The medical statement to verify
        model_name: Optional model name to use (defaults to config.model_names[0])

    Returns:
        Dictionary with verdict, topic, evidence, and confidence
    """
    # Initialize database
    db = Chroma(
        persist_directory=config.chroma_path, embedding_function=get_embeddings_func()
    )

    # Retrieve relevant chunks
    results = db.similarity_search_with_score(statement, k=config.k)

    if not results:
        return {"verdict": "UNVERIFIABLE", "topic": "unknown", "chunks_retrieved": 0}

    # Format context with topics
    context = format_context_with_topics(results)

    # Initialize Ollama LLM
    model_to_use = model_name or config.model_names[0]
    llm = OllamaLLM(
        model=model_to_use,
        temperature=0,
        base_url="http://localhost:11434",  # Default Ollama URL, adjust if needed
    )

    # Create prompt (using PromptTemplate for Ollama)
    prompt = PromptTemplate.from_template(FACT_CHECK_PROMPT)

    # Create chain
    chain = prompt | llm

    # Get response
    try:
        response = chain.invoke({"context": context, "statement": statement})

        # Extract JSON from response
        # Ollama might return the JSON wrapped in other text, so we try to extract it
        response_text = response if isinstance(response, str) else str(response)

        # Try to find JSON in the response
        import re

        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            result = json.loads(json_str)
        else:
            # If no JSON found, try to parse the entire response
            result = json.loads(response_text)

        # Store original verdict for logging/analysis
        result["original_verdict"] = result["verdict"]

        # Map UNSURE to TRUE to reduce false negatives
        if result["verdict"] == "UNSURE":
            result["verdict"] = "TRUE"

        # Add additional metadata
        result["chunks_retrieved"] = len(results)
        result["avg_relevance_score"] = sum(1 - score for _, score in results) / len(
            results
        )
        return result

    except json.JSONDecodeError as e:
        # Fallback if JSON parsing fails
        return {
            "verdict": "ERROR",
            "topic": "unknown",
            "chunks_retrieved": len(results),
            "raw_response": response_text if "response_text" in locals() else str(e),
            "error": str(e),
        }
    except Exception as e:
        # Handle other errors
        return {
            "verdict": "ERROR",
            "topic": "unknown",
            "chunks_retrieved": len(results),
            "error": str(e),
        }


def check_multiple_facts(statements: List[str]) -> List[Dict]:
    """
    Check multiple medical statements efficiently.

    Args:
        statements: List of medical statements to verify

    Returns:
        List of verification results
    """
    results = []
    for i, statement in enumerate(statements, 1):
        logger.info(f"Checking statement {i}/{len(statements)}: {statement[:100]}...")
        result = check_fact(statement)
        result["statement"] = statement
        results.append(result)

        # Log immediate result
        original = result.get("original_verdict", result["verdict"])
        if original != result["verdict"]:
            logger.info(
                f"Original: {original} -> Final: {result['verdict']} | Topic: {result['topic']}"
            )
        else:
            logger.info(f"Verdict: {result['verdict']} | Topic: {result['topic']}")

    return results


def print_detailed_results(results: List[Dict]):
    """
    Print detailed results in a formatted way and create plots.
    """
    logger.info("\n" + "=" * 80)
    logger.info("FACT CHECK RESULTS")
    logger.info("=" * 80)

    for i, result in enumerate(results, 1):
        logger.info(f"\n{i}. Statement: {result.get('statement', 'N/A')[:150]}...")
        logger.info(f"   Verdict: {result['verdict']}")
        logger.info(f"   Topic: {result['topic']}")
        logger.info(f"   Chunks Used: {result.get('chunks_retrieved', 'N/A')}")
        logger.info(f"   Avg Relevance: {result.get('avg_relevance_score', 0):.3f}")
        logger.info("-" * 40)

    # Summary statistics
    total = len(results)
    true_count = sum(1 for r in results if r["verdict"] == "TRUE")
    false_count = sum(1 for r in results if r["verdict"] == "FALSE")
    unverifiable_count = sum(1 for r in results if r["verdict"] == "UNVERIFIABLE")

    logger.info(f"\nSUMMARY:")
    logger.info(f"  Total statements: {total}")
    logger.info(f"  TRUE: {true_count} ({100 * true_count / total:.1f}%)")
    logger.info(f"  FALSE: {false_count} ({100 * false_count / total:.1f}%)")
    logger.info(
        f"  UNVERIFIABLE: {unverifiable_count} ({100 * unverifiable_count / total:.1f}%)"
    )

    # Always create plots
    plot_fact_check_results(results)


def export_results_to_json(results: List[Dict]):
    """
    Export results to a JSON file.
    """
    with open(config.export_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results exported to {config.export_file}")


def plot_fact_check_results(results: List[Dict]):
    """
    Create visualization plots for fact check results.
    """
    if not results:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Original verdict distribution pie chart (before UNSURE->TRUE mapping)
    original_verdict_counts = {}
    for result in results:
        original_verdict = result.get("original_verdict", result["verdict"])
        original_verdict_counts[original_verdict] = (
            original_verdict_counts.get(original_verdict, 0) + 1
        )

    if original_verdict_counts:
        axes[0, 0].pie(
            original_verdict_counts.values(),
            labels=original_verdict_counts.keys(),
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[0, 0].set_title(
            "Original Verdict Distribution\n(Before UNSURE→TRUE Mapping)"
        )

    # 2. Relevance score distribution
    relevance_scores = [
        r.get("avg_relevance_score", 0)
        for r in results
        if r.get("avg_relevance_score", 0) > 0
    ]
    if relevance_scores:
        axes[0, 1].hist(relevance_scores, bins=20, color="lightblue", alpha=0.7)
        axes[0, 1].set_title("Relevance Score Distribution")
        axes[0, 1].set_xlabel("Relevance Score")
        axes[0, 1].set_ylabel("Count")

    # 3. Processing time histogram
    processing_times = [
        r.get("processing_time", 0) for r in results if r.get("processing_time", 0) > 0
    ]
    if processing_times:
        axes[1, 0].hist(processing_times, bins=20, color="lightgreen", alpha=0.7)
        axes[1, 0].set_title("Processing Time Distribution")
        axes[1, 0].set_xlabel("Time (seconds)")
        axes[1, 0].set_ylabel("Frequency")

    # 4. Topic distribution (top 10)
    topic_counts = {}
    for result in results:
        topic = result.get("topic", "Unknown")
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    if topic_counts:
        # Get top 10 topics
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        topics, counts = zip(*top_topics)

        axes[1, 1].barh(topics, counts, color="coral")
        axes[1, 1].set_title("Top 10 Topics")
        axes[1, 1].set_xlabel("Count")

    plt.tight_layout()
    plt.savefig("fact_check_results.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Fact check results plot saved as 'fact_check_results.png'")


def main():
    """Main function - fact checker is intended for API use only."""
    logger.info("Fact checker is designed to be used via API calls.")
    logger.info(
        "Use the check_fact(statement) function directly or via the API endpoint."
    )
    logger.info(
        "For testing purposes, you can import and use the functions programmatically."
    )


if __name__ == "__main__":
    main()
