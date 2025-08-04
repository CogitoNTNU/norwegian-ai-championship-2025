"""Model registry for managing available embedding models."""

from typing import Dict, Type, Optional, Any
import yaml
from pathlib import Path

from .base import BaseEmbeddingModel
from .sentence_transformers import (
    SentenceTransformerModel,
    MatryoshkaModel,
    MedicalEmbeddingModel,
    E5Model,
)


class ModelRegistry:
    """Registry for embedding models."""

    # Default model configurations
    DEFAULT_MODELS = {
        # General-purpose models
        "all-MiniLM-L6-v2": {
            "class": SentenceTransformerModel,
            "params": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            },
        },
        "all-mpnet-base-v2": {
            "class": SentenceTransformerModel,
            "params": {
                "model_name": "sentence-transformers/all-mpnet-base-v2",
            },
        },
        "gte-base": {
            "class": SentenceTransformerModel,
            "params": {
                "model_name": "thenlper/gte-base",
            },
        },
        "gte-large": {
            "class": SentenceTransformerModel,
            "params": {
                "model_name": "thenlper/gte-large",
            },
        },
        "bge-base-en-v1.5": {
            "class": SentenceTransformerModel,
            "params": {
                "model_name": "BAAI/bge-base-en-v1.5",
            },
        },
        "bge-large-en-v1.5": {
            "class": SentenceTransformerModel,
            "params": {
                "model_name": "BAAI/bge-large-en-v1.5",
            },
        },
        "BGE-large-en-v1.5": {
            "class": SentenceTransformerModel,
            "params": {
                "model_name": "BAAI/bge-large-en-v1.5",
            },
        },
        "e5-base": {
            "class": E5Model,
            "params": {
                "model_name": "intfloat/e5-base",
            },
        },
        "e5-base-v2": {
            "class": E5Model,
            "params": {
                "model_name": "intfloat/e5-base-v2",
            },
        },
        # Matryoshka models
        "nomic-embed-text-v1.5": {
            "class": MatryoshkaModel,
            "params": {
                "model_name": "nomic-ai/nomic-embed-text-v1.5",
                "matryoshka_dims": [768, 512, 256, 128, 64],
                "trust_remote_code": True,
            },
        },
        # Medical models
        "pubmedbert-base-embeddings": {
            "class": MedicalEmbeddingModel,
            "params": {
                "model_name": "NeuML/pubmedbert-base-embeddings",
            },
        },
        "BioLORD-2023": {
            "class": MedicalEmbeddingModel,
            "params": {
                "model_name": "FremyCompany/BioLORD-2023",
            },
        },
        "Bio_ClinicalBERT": {
            "class": MedicalEmbeddingModel,
            "params": {
                "model_name": "emilyalsentzer/Bio_ClinicalBERT",
            },
        },
        # Stella models
        "stella-400M": {
            "class": SentenceTransformerModel,
            "params": {
                "model_name": "dunzhang/stella_en_400M_v5",
                "trust_remote_code": True,
            },
        },
        "stella-1.5B": {
            "class": SentenceTransformerModel,
            "params": {
                "model_name": "dunzhang/stella_en_1.5B_v5",
                "trust_remote_code": True,
            },
        },
        "stella-base": {
            "class": SentenceTransformerModel,
            "params": {
                "model_name": "infgrad/stella-base-en-v2",
            },
        },
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize model registry.

        Args:
            config_path: Path to custom configuration file
        """
        self.models = self.DEFAULT_MODELS.copy()
        self.config_path = config_path

        # Load custom configuration if provided
        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                config = yaml.safe_load(f)

            # Update models with custom configuration
            if "models" in config:
                for model_name, model_config in config["models"].items():
                    if model_name in self.models:
                        # Update existing model
                        self.models[model_name]["params"].update(model_config.get("params", {}))
                    else:
                        # Add new model
                        self.models[model_name] = model_config

    def register_model(self, name: str, model_class: Type[BaseEmbeddingModel], params: Dict[str, Any]) -> None:
        """
        Register a new model.

        Args:
            name: Model name
            model_class: Model class
            params: Model parameters
        """
        self.models[name] = {"class": model_class, "params": params}

    def get_model(self, name: str, **override_params) -> BaseEmbeddingModel:
        """
        Get a model instance.

        Args:
            name: Model name
            **override_params: Parameters to override

        Returns:
            Model instance
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found. Available models: {list(self.models.keys())}")

        model_config = self.models[name]
        model_class = model_config["class"]
        params = model_config["params"].copy()

        # Override parameters if provided
        params.update(override_params)

        return model_class(**params)

    def list_models(self) -> list:
        """List available models."""
        return list(self.models.keys())

    def get_model_info(self, name: str) -> Dict[str, Any]:
        """Get model information."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found")

        return {
            "name": name,
            "class": self.models[name]["class"].__name__,
            "params": self.models[name]["params"],
        }


# Global registry instance
_registry = None


def get_registry(config_path: Optional[str] = None) -> ModelRegistry:
    """Get or create the global registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(config_path)
    return _registry


def get_embedding_model(model_name: str, config_path: Optional[str] = None, **kwargs) -> BaseEmbeddingModel:
    """
    Get an embedding model instance.

    Args:
        model_name: Name of the model
        config_path: Path to configuration file
        **kwargs: Additional parameters

    Returns:
        Embedding model instance
    """
    registry = get_registry(config_path)
    return registry.get_model(model_name, **kwargs)
