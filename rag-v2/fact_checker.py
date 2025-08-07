from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from loguru import logger
from embeddings import get_embeddings_func
from get_config import config
import json
import re
from typing import Dict, List, Tuple
import argparse


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


def format_context_with_topics(results: List[Tuple]) -> str:
    """
    Format the search results to include topic information prominently.
    """
    formatted_chunks = []
    for i, (doc, score) in enumerate(results, 1):
        topic = doc.metadata.get("topic", "unknown")
        
        formatted_chunk = f"""
Topic: {topic}:
{doc.page_content}
"""
        formatted_chunks.append(formatted_chunk)
    
    return "\n".join(formatted_chunks)


def check_fact(statement: str, k: int = 5, model_name: str = "cogito:32b") -> Dict:
    """
    Check if a medical statement is true or false based on the knowledge base.
    
    Args:
        statement: The medical statement to verify
        k: Number of relevant chunks to retrieve
        model_name: Ollama model to use for fact checking (default: cogito:8b)
    
    Returns:
        Dictionary with verdict, topic, evidence, and confidence
    """
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
            "chunks_retrieved": 0
        }
    
    # Format context with topics
    context = format_context_with_topics(results)
    logger.debug(context)
    
    # Initialize Ollama LLM
    llm = OllamaLLM(
        model=model_name,
        temperature=0,
        base_url="http://localhost:11434"  # Default Ollama URL, adjust if needed
    )
    
    # Create prompt (using PromptTemplate for Ollama)
    prompt = PromptTemplate.from_template(FACT_CHECK_PROMPT)
    
    # Create chain
    chain = prompt | llm
    
    # Get response
    try:
        response = chain.invoke({
            "context": context,
            "statement": statement
        })
        
        # Extract JSON from response
        # Ollama might return the JSON wrapped in other text, so we try to extract it
        response_text = response if isinstance(response, str) else str(response)
        logger.debug(f"Response from llm:\n\n{response_text}")
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            result = json.loads(json_str)
        else:
            # If no JSON found, try to parse the entire response
            result = json.loads(response_text)
        
        result["chunks_retrieved"] = len(results)
        result["avg_relevance_score"] = sum(1-score for _, score in results) / len(results)
        return result
        
    except json.JSONDecodeError as e:
        # Fallback if JSON parsing fails
        return {
            "verdict": "ERROR",
            "topic": "unknown",
            "evidence": "Failed to parse response",
            "confidence": "LOW",
            "chunks_retrieved": len(results),
            "raw_response": response_text if 'response_text' in locals() else str(e),
            "error": str(e)
        }
    except Exception as e:
        # Handle other errors
        return {
            "verdict": "ERROR",
            "topic": "unknown",
            "evidence": f"Error during processing: {str(e)}",
            "confidence": "LOW",
            "chunks_retrieved": len(results),
            "error": str(e)
        }


def check_multiple_facts(statements: List[str], k: int = 5, model_name: str = "cogito:8b") -> List[Dict]:
    """
    Check multiple medical statements efficiently.
    
    Args:
        statements: List of medical statements to verify
        k: Number of relevant chunks to retrieve per statement
        model_name: Ollama model to use (default: cogito:8b)
    
    Returns:
        List of verification results
    """
    results = []
    for i, statement in enumerate(statements, 1):
        if i < 6:
            continue
        print(f"\nChecking statement {i}/{len(statements)}: {statement[:100]}...")
        result = check_fact(statement, k, model_name)
        result["statement"] = statement
        results.append(result)
        
        # Print immediate result
        print(f"  Verdict: {result['verdict']} | Topic: {result['topic']} | Confidence: {result['confidence']}")
    
    return results


def print_detailed_results(results: List[Dict]):
    """
    Print detailed results in a formatted way.
    """
    print("\n" + "="*80)
    print("FACT CHECK RESULTS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Statement: {result.get('statement', 'N/A')[:150]}...")
        print(f"   Verdict: {result['verdict']}")
        print(f"   Topic: {result['topic']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Evidence: {result['evidence'][:200]}...")
        print(f"   Chunks Used: {result.get('chunks_retrieved', 'N/A')}")
        print(f"   Avg Relevance: {result.get('avg_relevance_score', 0):.3f}")
        print("-"*40)
    
    # Summary statistics
    total = len(results)
    true_count = sum(1 for r in results if r['verdict'] == 'TRUE')
    false_count = sum(1 for r in results if r['verdict'] == 'FALSE')
    
    print(f"\nSUMMARY:")
    print(f"  Total statements: {total}")
    print(f"  TRUE: {true_count} ({100*true_count/total:.1f}%)")
    print(f"  FALSE: {false_count} ({100*false_count/total:.1f}%)")


def export_results_to_json(results: List[Dict], filename: str = "fact_check_results.json"):
    """
    Export results to a JSON file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults exported to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Medical Fact Checker")
    parser.add_argument("statement", nargs="?", help="Statement to fact-check")
    parser.add_argument("--file", "-f", help="File containing statements (one per line)")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--model", default="cogito:8b", help="Ollama model to use (default: cogito:8b)")
    parser.add_argument("--export", "-e", help="Export results to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed results")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama server URL (default: http://localhost:11434)")
    
    args = parser.parse_args()
    
    # Collect statements to check
    statements = []
    
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            statements = [line.strip() for line in f if line.strip()]
    elif args.statement:
        statements = [args.statement]
    else:
        # Interactive mode
        print("Medical Fact Checker - Interactive Mode")
        print("Enter statements to check (empty line to finish):")
        while True:
            statement = input("> ").strip()
            if not statement:
                break
            statements.append(statement)
    
    if not statements:
        print("No statements to check.")
        return
    
    # Check facts
    results = check_multiple_facts(statements, k=args.k, model_name=args.model)
    
    # Display results
    if args.verbose:
        print_detailed_results(results)
    else:
        # Simple output
        print("\n" + "="*60)
        print("RESULTS:")
        print("="*60)
        for i, result in enumerate(results, 1):
            statement_preview = result.get('statement', 'N/A')[:80]
            if len(result.get('statement', '')) > 80:
                statement_preview += "..."
            print(f"{i}. {statement_preview}")
            print(f"   → {result['verdict']} ({result['topic']}) - Confidence: {result['confidence']}")
    
    # Export if requested
    if args.export:
        export_results_to_json(results, args.export)


if __name__ == "__main__":
    main()
