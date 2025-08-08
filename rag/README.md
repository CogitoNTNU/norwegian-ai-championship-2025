# Emergency Healthcare RAG

To run rocm:

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
ollama pull cogito:32b

# Also pull the embeddings model
ollama pull mxbai-embed-large

ollama serve &

uv run populate_db.py
```

## Usage

### API Server

```bash
uv run api
```

The API will be available at `http://localhost:8000`

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

## Test local LLM with the test data

```bash
uv run data_tester.py
```

## Available Models

### Embeddings

- `mxbai-embed-large`

### LLMs (via Ollama)

- `cogito:32b` - Default, run `ollama pull cogito:32b`
- `cogito:14b` - Run `ollama pull cogito:14b`

Remember to use the correct LLM settings in the config file!

## Configuration

Most things can be configured in the `config.yaml` file.

## Validation

Submit for validation against competition server:

```bash
uv run validate
```

## Performance

- Response time: ~2-5 seconds per statement with AMD rx 7900XTX
- Completely offline - no cloud APIs
