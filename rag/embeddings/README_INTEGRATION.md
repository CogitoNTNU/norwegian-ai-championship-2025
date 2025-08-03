# Embeddings Integration with RAG Pipeline

## Overview
The embeddings system has been successfully integrated with the RAG pipeline for the medical healthcare competition. This integration allows for flexible embedding model selection and comparison.

## Key Components

### 1. Document Manager (`managers/document_manager.py`)
- Manages shared document chunks across different embedding models
- Caches chunks to avoid redundant processing
- Handles document loading from `data/topics/`

### 2. Embeddings Document Store (`rag-pipeline/document_store_embeddings.py`)
- Uses configurable embedding models from the embeddings module
- Integrates with FAISS for vector storage
- Supports model-specific indices while sharing document chunks

### 3. Embeddings RAG Pipeline (`rag-pipeline/rag_pipeline_embeddings.py`)
- Complete RAG pipeline with configurable embeddings
- Integrates with LLM for medical statement classification
- Returns `{statement_is_true: 0/1, statement_topic: 0-114}`

### 4. Evaluation Template (`Rag-evaluation/src/templates/embeddings_rag.py`)
- Standard evaluation template for the RAG evaluation framework
- Allows easy model switching for comparison

## Usage

### Basic Pipeline Usage
```python
from rag_pipeline.rag_pipeline_embeddings import EmbeddingsRAGPipeline

# Initialize with desired embedding model
pipeline = EmbeddingsRAGPipeline(
    embedding_model="all-MiniLM-L6-v2",  # or any supported model
    llm_model="qwen3:8b",
    top_k_retrieval=5
)

# Setup with data
pipeline.setup(
    topics_dir="data/topics",
    topics_json="data/topics.json"
)

# Make prediction
is_true, topic = pipeline.predict("Diabetes affects blood sugar levels")
```

### Model Comparison
```bash
cd rag/embeddings/scripts
python compare_embeddings.py
```

This will evaluate multiple embedding models and show:
- Binary classification accuracy
- Topic classification accuracy
- Combined accuracy
- Inference times

### Testing Integration
```bash
cd rag/embeddings/scripts
python test_integration.py
```

## Available Models
- `all-MiniLM-L6-v2` - Fast baseline (384 dim)
- `all-mpnet-base-v2` - Higher quality (768 dim)
- `multi-qa-MiniLM-L6-cos-v1` - Q&A optimized (384 dim)
- `e5-small-v2` - E5 family model (384 dim)
- `bge-small-en-v1.5` - BGE model (384 dim)
- `gte-small` - GTE model (384 dim)
- Norwegian models (when available)

## Performance Notes
- Document chunks are cached and shared across models
- Each model maintains its own FAISS index
- Indices are cached for faster subsequent runs
- The system automatically detects when documents change

## Next Steps
1. Run the comparison script to find the best performing model
2. Update the main pipeline to use the optimal model
3. Consider fine-tuning on medical domain if needed