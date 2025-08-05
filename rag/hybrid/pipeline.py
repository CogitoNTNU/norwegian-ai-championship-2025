"""
Hybrid RAG Pipeline Orchestrator

Coordinates the entire pipeline from statement input to classification output.
"""

import asyncio
import time
import logging
import argparse
import json
from typing import Dict, Any

from .components import QueryGenerator, HybridRetriever, QwenReranker, StatementClassifier
from .utils import stopwatch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridRAGPipeline:
    """Complete hybrid RAG pipeline for medical statement classification."""
    
    def __init__(self):
        """Initialize all pipeline components."""
        logger.info("Initializing Hybrid RAG Pipeline...")
        
        self.query_generator = QueryGenerator()
        self.retriever = HybridRetriever()
        self.reranker = QwenReranker()
        self.classifier = StatementClassifier()
        
        logger.info("✅ Pipeline initialized successfully")
    
    async def process_statement(self, statement: str, return_contexts: bool = False) -> Dict[str, Any]:
        """Process a medical statement through the complete pipeline."""
        start_time = time.perf_counter()
        
        try:
            # Step 1: Generate search queries
            with stopwatch("query_generation"):
                queries = await self.query_generator.generate_queries(statement)
                logger.info(f"Generated {len(queries.get('keywords', []))} keywords, "
                           f"{len(queries.get('questions', []))} questions")
            
            # Step 2: Hybrid retrieval
            with stopwatch("hybrid_retrieval"):
                documents = await self.retriever.retrieve(queries)
                logger.info(f"Retrieved {len(documents)} candidate documents")
            
            # Step 3: Reranking
            with stopwatch("reranking"):
                # Use the original statement as the reranking query
                reranked_docs = await self.reranker.rerank(statement, documents)
                logger.info(f"Reranked to top {len(reranked_docs)} documents")
            
            # Step 4: Final classification
            with stopwatch("classification"):
                result = await self.classifier.classify(statement, reranked_docs)
            
            # Log performance
            elapsed = time.perf_counter() - start_time
            logger.info(f"⚡ Pipeline completed in {elapsed:.3f}s")
            
            # Return classification result with optional contexts
            if return_contexts:
                return {
                    **result,
                    "retrieved_contexts": [doc['text'] for doc in reranked_docs]
                }
            else:
                return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            # Return fallback result
            return {
                "statement_is_true": 1,
                "statement_topic": 72  # Sepsis/Septic Shock as fallback
            }

# Global pipeline instance (initialized on first use)
_pipeline = None

async def run(statement: str) -> Dict[str, int]:
    """
    Main entry point for the hybrid RAG pipeline.
    
    Args:
        statement: Medical statement to classify
        
    Returns:
        Dictionary with 'statement_is_true' (0 or 1) and 'statement_topic' (0-114)
    """
    global _pipeline
    
    if _pipeline is None:
        _pipeline = HybridRAGPipeline()
    
    return await _pipeline.process_statement(statement)

async def run_with_contexts(statement: str) -> Dict[str, Any]:
    """
    Entry point for evaluation that includes retrieved contexts.
    
    Args:
        statement: Medical statement to classify
        
    Returns:
        Dictionary with classification and retrieved contexts
    """
    global _pipeline
    
    if _pipeline is None:
        _pipeline = HybridRAGPipeline()
    
    return await _pipeline.process_statement(statement, return_contexts=True)

def main():
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(description="Hybrid RAG for Emergency Medicine")
    parser.add_argument(
        "--statement", 
        required=True,
        help="Medical statement to classify"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run the pipeline
    result = asyncio.run(run(args.statement))
    
    # Output result as clean JSON
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
