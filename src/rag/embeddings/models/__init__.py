"""Embedding models module."""

from .base import BaseEmbeddingModel
from .model_registry import ModelRegistry, get_embedding_model

__all__ = ["BaseEmbeddingModel", "ModelRegistry", "get_embedding_model"]
