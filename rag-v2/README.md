# Emergency Healthcare RAG

to run:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTORCH_ROCM_ARCH="gfx1100" ollama serve
```

Simple RAG pipeline for medical statement classification using embeddings and FAISS.

## Overview

This system classifies medical statements by:

- **Binary classification**: Is the statement true (1) or false (0)?
- **Topic classification**: Which of 115 medical topics does it belong to?

Uses embeddings-based retrieval with FAISS vector search and local LLMs via Ollama.

## Setup

```bash
# Install dependencies
uv sync

# Install Ollama and pull an LLM
ollama pull cogito:8b
```

## Usage

### API Server

```bash
uv run api
```

The API will be available at `http://localhost:8000`

### Command Line

```bash
# Basic usage
uv run python run_rag.py --statement "Diabetes is a chronic metabolic disorder"

# With specific models
uv run python run_rag.py \
    --statement "Aspirin is used to prevent blood clots" \
    --embedding pubmedbert-base-embeddings \
    --llm cogito:8b

# Test multiple statements from training data
uv run python test_pipeline.py --n 15 --embedding pubmedbert-base-embeddings --llm cogito:8b

# Test with verbose output
uv run python test_pipeline.py --n 5 --verbose --embedding gte-base --llm llama3.2:latest
```

### API Endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"statement": "Heart attacks require immediate medical attention"}'
```

Response:

```json
{
  "statement_is_true": 1,
  "statement_topic": 2
}
```

## Available Models

### Embeddings

- `pubmedbert-base-embeddings` - Medical-focused (has existing index)
- `gte-base` - General purpose (has existing index)
- `all-MiniLM-L6-v2` - Fast and lightweight
- `all-mpnet-base-v2` - High quality general purpose
- `BioLORD-2023` - Biomedical specialist

### LLMs (via Ollama)

- `cogito:8b` - Default
- `cogito:14b` - Run `ollama pull cogito:14b`
- `qwen3:8b`
- `llama3.2:latest`

## Configuration

Set environment variables to change default models:

```bash
export EMBEDDING_MODEL=gte-base
export LLM_MODEL=cogito:8b
```

## Validation

Submit for validation against competition server:

```bash
uv run validate
```

## Structure

```
rag/
├── api.py              # FastAPI server
├── model.py            # Core prediction function  
├── run_rag.py          # CLI interface
├── embeddings/         # Embedding models and managers
├── rag-pipeline/       # RAG pipeline implementation
├── indices/            # FAISS vector indices
└── data/              # Medical topics and training data
```

## Performance

- Response time: ~2-5 seconds per statement
- Memory usage: < 8GB RAM (excluding LLM)
- Completely offline - no cloud APIs
