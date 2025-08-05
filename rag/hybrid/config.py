"""
Configuration for Hybrid RAG System
"""

import json
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "hybrid_artifacts"

# Ensure cache directory exists
CACHE_DIR.mkdir(exist_ok=True)

# Model configurations
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
OLLAMA_MODEL = "cogito:3b"

# Retrieval parameters
K_SPARSE = 20
K_DENSE = 60
K_FINAL = 10  # Top k after reranking

# Performance optimizations for speed
USE_LIGHT_RERANKER = True  # Use MixedBread instead of Qwen3 for speed
CANDIDATES_TO_RERANK = 20  # Further reduced for better precision
CLASSIFICATION_MAX_CHUNKS = 3  # Limit evidence to 3 best chunks
CHUNK_MAX_TOKENS = 120  # Shorter chunks for better signal

# Fast reranker model
MIXEDBREAD_RERANKER = "mixedbread-ai/mxbai-rerank-base-v1"

# Generation limits for speed
QUERY_GENERATION_MAX_TOKENS = 128  # Increased for better query diversity
CLASSIFICATION_MAX_TOKENS = 64  # Doubled from 32 for better topic classification

# Document processing paths - Phase 2: Rich Document Chunks
DOCUMENT_CHUNKS_DIR = DATA_DIR / "processed" / "document_chunks"
CHUNK_METADATA_FILE = DOCUMENT_CHUNKS_DIR / "chunk_metadata.json"
TOPIC_MAPPING_FILE = DOCUMENT_CHUNKS_DIR / "topic_mapping.json"

# Legacy paths (for backward compatibility)
STATEMENTS_DIR = DATA_DIR / "processed" / "combined" / "statements"
ANSWERS_DIR = DATA_DIR / "processed" / "combined" / "answers"
TOPICS_FILE = DATA_DIR / "topics.json"

def load_topic_mapping():
    """Load the emergency medicine topic mapping."""
    with open(TOPICS_FILE, 'r') as f:
        return json.load(f)

def get_cache_path(filename: str) -> Path:
    """Get path for cached file."""
    return CACHE_DIR / filename

# Cache file paths - Phase 2: Document Chunks
BM25_CACHE = get_cache_path("bm25_doc_chunks.pkl")
FAISS_CACHE = get_cache_path("faiss_doc_chunks.index")
META_CACHE = get_cache_path("metadata_doc_chunks.pkl")

# Environment-specific settings
MAX_CONTENT_LENGTH = 4000  # For embeddings and reranking
BATCH_SIZE = 32
DEVICE_PREFERENCE = ["mps", "cpu"]  # Apple Silicon preference
