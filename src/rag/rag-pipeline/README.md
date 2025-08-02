# Emergency Healthcare RAG Pipeline

A minimal viable RAG (Retrieval-Augmented Generation) solution for the Norwegian AI Championship 2025 Emergency Healthcare challenge.

## Architecture

```
Medical Statement → Vector Search → LLM Classification → {is_true, topic_id}
                     ↓
                Medical Documents
                (115 topics from competition data)
```

## Components

- **DocumentStore** (`document_store.py`): Processes medical articles into searchable chunks using FAISS
- **LLMClient** (`llm_client.py`): Interfaces with local Ollama LLM (Qwen3 8B)
- **RAGPipeline** (`rag_pipeline.py`): Combines retrieval + LLM for classification
- **Classifier** (`classifier.py`): Main interface matching competition API
- **Setup Scripts** (`setup.py`, `setup_uv.py`): Automated setup scripts
- **Integration** (`integration.py`): Connects to competition framework

## Quick Start

### 1. Install Dependencies (UV Method - Recommended)

```bash
cd src/rag
uv sync
```

### 2. Run Complete Setup (UV)

```bash
uv run python setup_uv.py
```

**OR Traditional pip method:**

```bash
pip install -r requirements.txt
python setup.py
```

This will:

- Install Ollama (local LLM server)
- Download Qwen3 8B model (~5GB)
- Process medical documents into vector index
- Test the classifier
- Run evaluation on training samples

### 3. Integrate with Competition

```bash
python integration.py
```

This replaces the competition's `model.py` with RAG-powered version.

### 4. Test Competition Framework

```bash
cd ../../DM-i-AI-2025/emergency-healthcare-rag
python example.py
python api.py
```

## Technical Details

### Performance Constraints

- **Speed**: \<5 seconds per statement (achieved via quantized 8B model)
- **Memory**: \<24GB VRAM (Qwen3 8B uses ~5GB)
- **Offline**: No cloud APIs (everything runs locally)

### Model Pipeline

1. **Document Processing**: 115 medical topics chunked into ~300-char segments
1. **Embedding**: `all-MiniLM-L6-v2` for semantic search
1. **Retrieval**: Top-3 most relevant chunks via FAISS cosine similarity
1. **Classification**: Qwen3 8B with structured prompting
1. **Output**: Binary classification (0/1) + Topic ID (0-114)

### Fallback Strategy

If RAG fails, system falls back to keyword matching baseline to ensure reliability.

## File Structure

```
src/rag/
├── requirements.txt      # Python dependencies
├── document_store.py     # Vector database + chunking
├── llm_client.py        # Ollama LLM interface
├── rag_pipeline.py      # Main RAG logic
├── classifier.py        # Competition API interface
├── setup.py            # Automated setup
├── integration.py      # Competition integration
├── medical_index.*     # Generated vector index (after setup)
└── README.md          # This file
```

## Development Notes

### Testing Individual Components

```python
# Test document store
from document_store import DocumentStore

store = DocumentStore()
store.load_medical_documents("../path/to/topics", "../path/to/topics.json")
store.build_index()
results = store.search("testicular torsion")

# Test LLM client
from llm_client import LocalLLMClient

client = LocalLLMClient()
client.ensure_model_available()
result = client.classify_statement("test statement", "context")

# Test full pipeline
from classifier import predict

result = predict("Testicular torsion is a surgical emergency")
```

### Performance Tuning

- Adjust `top_k_retrieval` in RAGPipeline (default: 3)
- Modify chunk size in DocumentStore (default: 300 chars)
- Change LLM temperature for consistency vs creativity
- Use different embedding models for domain specificity

### Common Issues

1. **Ollama not starting**: Ensure Docker-like permissions for Ollama
1. **Model download fails**: Check internet connection and disk space
1. **Out of memory**: Use smaller model (llama3.1:3b) or increase swap
1. **Slow inference**: Reduce context length or use faster embeddings

## Next Steps for Improvement

1. **Fine-tuning**: Fine-tune embeddings on medical domain
1. **Prompt Engineering**: Optimize LLM prompts for medical reasoning
1. **Ensemble**: Combine multiple models for robustness
1. **Caching**: Cache frequent queries for speed
1. **Preprocessing**: Better medical text normalization

The system is designed for hackathon speed - get something working quickly, then iterate!
