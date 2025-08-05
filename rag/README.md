# Emergency Healthcare RAG

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

### GPU Setup with RunPod (Optional)

To use a GPU server for LLM inference while running RAG locally:

#### 1. Setup RunPod GPU Server

1. **SSH into your RunPod instance**

1. **Install and start Ollama**:

   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh

   # Start Ollama server (keep this running)
   ollama serve
   ```

1. **In a new terminal, pull your model**:

   ```bash
   ollama pull cogito:14b  # or cogito:8b for smaller model
   ollama list            # Verify model is downloaded
   ```

1. **Create Pinggy tunnel for Ollama** (port 11434):

   ```bash
   ssh -p 443 -R0:localhost:11434 a.pinggy.io
   ```

   Keep this terminal open and copy the HTTPS URL (e.g., `https://xxxxx.a.free.pinggy.link`)

#### 2. Configure Local Environment

```bash
# Set the Ollama host to your RunPod pinggy URL
export OLLAMA_HOST=https://xxxxx.a.free.pinggy.link

# Optional: Set environment variables for Mac compatibility
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
```

#### 3. Start RAG Server Locally

```bash
# Navigate to rag directory
cd rag/

# Start the server
uv run api
```

The server will now use:

- **Local**: Embeddings, document retrieval, FAISS indexing
- **GPU (RunPod)**: LLM inference for classification

````

## Usage

### API Server

```bash
uv run api
````

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
- `Bio_ClinicalBERT` - Clinical text specialist

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

For external LLM server:

```bash
export OLLAMA_HOST=https://your-external-server.com
# The LLM_MODEL will be ignored when OLLAMA_HOST is set
```

## Testing and Validation

### Local Testing

1. **Test the API locally**:

   ```bash
   # Test endpoints
   curl http://localhost:8000/
   curl http://localhost:8000/api

   # Test prediction
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"statement": "Aspirin is used to treat heart attacks"}'
   ```

1. **Run test suite**:

   ```bash
   # Test with sample statements
   uv run python test_pipeline.py --n 10

   # Test with external LLM
   export OLLAMA_HOST=https://your-runpod-pinggy.a.free.pinggy.link
   uv run python test_pipeline.py --n 10
   ```

### Competition Validation

#### Method 1: Using Pinggy Tunnel

1. **Create tunnel for your local server** (port 8000):

   ```bash
   ssh -p 443 -R0:localhost:8000 free.pinggy.io
   ```

   Copy the HTTPS URL (e.g., `https://yyyyy.a.free.pinggy.link`)

1. **Submit to competition**:

   - Go to https://cases.ainm.no/
   - Navigate to Emergency Healthcare RAG
   - Enter your Pinggy URL with `/predict` endpoint: `https://yyyyy.a.free.pinggy.link/predict`
   - Enter your competition token
   - Submit for validation

1. **Monitor logs** (in another terminal):

   ```bash
   tail -f logs/api.log
   ```

#### Method 2: Using Built-in Script

```bash
# Set your competition token
export EVAL_API_TOKEN="your-token-here"

# Run validation
uv run validate
```

### Important Notes

- **Pinggy tunnels expire after 60 minutes** - create new ones as needed
- **Keep both terminals open**: One for the server, one for the pinggy tunnel
- **For GPU setup**: You need TWO pinggy tunnels:
  - One for Ollama on RunPod (port 11434)
  - One for your local RAG server (port 8000)

## Troubleshooting

### Common Issues

1. **"Server disconnected without sending a response"**

   - RunPod Ollama server is down or pinggy URL expired
   - Solution: Create new pinggy tunnel on RunPod and update OLLAMA_HOST

1. **Multiprocessing errors on Mac**

   - Set environment variables:
     ```bash
     export TOKENIZERS_PARALLELISM=false
     export OMP_NUM_THREADS=1
     ```

1. **Empty response from server**

   - Check server is running: `lsof -i :8000`
   - Restart with proper environment variables

1. **Pinggy connection failures**

   - Ensure server is running before creating tunnel
   - Use `0.0.0.0` as host when starting server

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

- Response time: ~2-5 seconds per statement (local), ~0.7s with GPU
- Memory usage: < 8GB RAM (excluding LLM)
- Completely offline - no cloud APIs
- GPU acceleration: Supported via external Ollama server
