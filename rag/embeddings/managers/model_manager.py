"""Model manager for downloading, caching, and switching between models."""

import os
from pathlib import Path
from typing import Dict, Optional, Any
import json
import shutil
from datetime import datetime

from ..models import get_embedding_model, BaseEmbeddingModel


class ModelManager:
    """Manages embedding models - downloading, caching, and switching."""

    def __init__(
        self, cache_dir: Optional[str] = None, config_path: Optional[str] = None
    ):
        """
        Initialize model manager.

        Args:
            cache_dir: Directory for caching models
            config_path: Path to configuration file
        """
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.cache/embeddings"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = config_path
        self.current_model: Optional[BaseEmbeddingModel] = None
        self.current_model_name: Optional[str] = None

        # Model metadata cache
        self.metadata_file = self.cache_dir / "model_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata from cache."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self) -> None:
        """Save model metadata to cache."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def load_model(
        self, model_name: str, force_download: bool = False, **kwargs
    ) -> BaseEmbeddingModel:
        """
        Load a model, downloading if necessary.

        Args:
            model_name: Name of the model
            force_download: Force re-download
            **kwargs: Additional model parameters

        Returns:
            Loaded model
        """
        # Unload current model if different
        if self.current_model and self.current_model_name != model_name:
            self.unload_current_model()

        # Check if already loaded
        if self.current_model_name == model_name:
            return self.current_model

        # Get model with cache directory
        model = get_embedding_model(
            model_name,
            config_path=self.config_path,
            cache_folder=str(self.cache_dir / "models"),
            **kwargs,
        )

        # Update metadata
        self.metadata[model_name] = {
            "last_loaded": datetime.now().isoformat(),
            "dimension": model.get_dimension(),
            "max_seq_length": model.get_max_seq_length(),
            "supports_matryoshka": model.supports_matryoshka(),
        }
        self._save_metadata()

        # Set as current model
        self.current_model = model
        self.current_model_name = model_name

        return model

    def unload_current_model(self) -> None:
        """Unload the current model to free memory."""
        if self.current_model:
            self.current_model.unload_model()
            self.current_model = None
            self.current_model_name = None

    def switch_model(self, model_name: str, **kwargs) -> BaseEmbeddingModel:
        """
        Switch to a different model.

        Args:
            model_name: Name of the model to switch to
            **kwargs: Additional model parameters

        Returns:
            New model
        """
        return self.load_model(model_name, **kwargs)

    def get_current_model(self) -> Optional[BaseEmbeddingModel]:
        """Get the currently loaded model."""
        return self.current_model

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a model.

        Args:
            model_name: Model name

        Returns:
            Model information
        """
        if model_name in self.metadata:
            return self.metadata[model_name]

        # Load model to get info
        self.load_model(model_name)
        return self.metadata[model_name]

    def list_cached_models(self) -> list:
        """List models that have been cached."""
        return list(self.metadata.keys())

    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """
        Clear model cache.

        Args:
            model_name: Specific model to clear, or None for all
        """
        if model_name:
            # Clear specific model
            model_dir = self.cache_dir / "models" / model_name.replace("/", "_")
            if model_dir.exists():
                shutil.rmtree(model_dir)

            if model_name in self.metadata:
                del self.metadata[model_name]
                self._save_metadata()
        else:
            # Clear all models
            models_dir = self.cache_dir / "models"
            if models_dir.exists():
                shutil.rmtree(models_dir)

            self.metadata = {}
            self._save_metadata()

    def estimate_memory_usage(self, model_name: str) -> Dict[str, Any]:
        """
        Estimate memory usage for a model.

        Args:
            model_name: Model name

        Returns:
            Memory usage estimate
        """
        info = self.get_model_info(model_name)
        dimension = info.get("dimension", 768)

        # Rough estimates based on model architecture
        base_memory_mb = 50  # Base overhead

        # Estimate based on dimension (very rough)
        if dimension <= 384:
            model_memory_mb = 100  # Small models
        elif dimension <= 768:
            model_memory_mb = 400  # Base models
        else:
            model_memory_mb = 1200  # Large models

        return {
            "estimated_memory_mb": base_memory_mb + model_memory_mb,
            "dimension": dimension,
            "note": "This is a rough estimate",
        }
