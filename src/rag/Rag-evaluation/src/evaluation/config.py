from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

EVALUATE_METHODS = [
    "simple_rag",
    "hybrid_rag",
    "hyde",
    "contextual_retriever",
    # "graph_rag_graph_retriever",
    # "graph_rag_hybrid_retriever",
    # "graph_rag_hybrid",
    # "graph_rag",
    "query_expansion_with_rrf",
    "query_rewrite_rag",
    "query_rewrite",
    "rrf",
    "semantic_chunker",
    "step_back_prompt",
    "step_back_prompting",
]
RUN_TAG = "initial_test"
RAGAS_METRICS = [
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
]
