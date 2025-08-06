from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

EVALUATE_METHODS = [
    # "topic_first_rag",  # Topic-first sequential classification
    # "healthcare_rag",
    # "hybrid_rag_dual_process",  # Dual process parallel classification
    "hybrid_rag_apple_silicon",  # Our new hybrid RAG system
    # "biobert_rag",  # BioBERT Hybrid + NLI reranking
    # "biobert_topic_bm25",  # BioBERT Topic-classified BM25 + NLI
    # "optimized_smart_rag",  # Optimized SmartRAG with pre-computed topic indexes
    # "simple_rag",
    # "embeddings_rag",  # RAG with configurable embeddings from rag-pipeline
    # "graph_rag",  # Graph-based RAG with simulated medical knowledge graph
    # "hyde",  # HyDE (Hypothetical Document Embeddings) RAG
    # "query_rewrite_rag",  # Query rewriting for enhanced retrieval
    # "pocketflow_rag",  # Full RAG with LLM generation
    # "hybrid_rag",
    # "contextual_retriever",
    # "graph_rag_graph_retriever",
    # "graph_rag_hybrid_retriever",
    # "graph_rag_hybrid",
    # "query_expansion_with_rrf",
    # "query_rewrite",
    # "rrf",
    # "semantic_chunker",
    # "step_back_prompt",
    # "step_back_prompting"
]
RUN_TAG = "healthcare_rag_bm25_no_leakage_qwen8b"
RAGAS_METRICS = [
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
]
