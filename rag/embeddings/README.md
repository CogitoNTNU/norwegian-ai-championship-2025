# Embeddings System for Emergency Healthcare RAG

This embeddings system provides a flexible and extensible framework for managing, testing, and fine-tuning embedding models for the emergency healthcare classification task.

## Features

- **Multiple Embedding Models**: Support for various sentence-transformer models including general-purpose and medical-specific models
- **Easy Model Switching**: Configuration-based model selection without code changes
- **Fine-tuning Support**: Infrastructure for domain adaptation on medical data
- **Benchmarking**: Compare models on speed, accuracy, and memory usage
- **FAISS Integration**: Seamless integration with existing vector database

## Quick Start

```python
# Add to your imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Add src/ to path

from rag.embeddings.models import get_embedding_model

# Load a model
model = get_embedding_model("nomic-embed-text-v1.5")

# Encode text
embeddings = model.encode(["Medical statement here"])

# Get model info
print(f"Model: {model.get_model_name()}")
print(f"Dimensions: {model.get_dimension()}")
```

## Available Models

### General-Purpose Models

- `nomic-embed-text-v1.5`: 137M params, Matryoshka support, 8192 token context
- `gte-base`: Alibaba's General Text Embeddings
- `bge-base-en-v1.5`: BAAI General Embedding
- `all-mpnet-base-v2`: High-quality general embeddings
- `all-MiniLM-L6-v2`: Fast and efficient (current default)

### Medical-Specific Models

- `pubmedbert-base-embeddings`: Fine-tuned on medical literature
- `MedCPT`: Contrastive pre-trained on PubMed logs
- `BioLORD-2023`: Biomedical embeddings with knowledge graph insights

## Configuration

Edit `configs/models.yaml` to set the default model and configure model parameters:

```yaml
default_model: "nomic-embed-text-v1.5"
models:
  nomic-embed-text-v1.5:
    dimensions: 768
    max_seq_length: 8192
    matryoshka_dims: [768, 512, 256, 128]
```

## Fine-tuning

Fine-tune models on your medical data:

```bash
cd src/rag/embeddings
python scripts/fine_tune.py --base-model gte-base --epochs 10
```

## Benchmarking

Compare models on your dataset:

```bash
cd src/rag/embeddings
python scripts/benchmark_models.py --models all
```

## Integration with RAG Pipeline

The system is designed to integrate seamlessly with the existing DocumentStore:

```python
# In document_store.py
from rag.embeddings.models import get_embedding_model


class DocumentStore:
    def __init__(self, embedding_model: str = "nomic-embed-text-v1.5"):
        self.embedding_model = get_embedding_model(embedding_model)
```

## Directory Structure

```
embeddings/
├── models/          # Model implementations
├── managers/        # Model and index management
├── fine_tuning/     # Fine-tuning infrastructure
├── configs/         # Configuration files
├── utils/           # Utility functions
├── scripts/         # Executable scripts
└── tests/           # Unit tests
```
