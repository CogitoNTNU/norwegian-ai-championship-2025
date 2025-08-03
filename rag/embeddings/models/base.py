"""Base interface for all embedding models."""

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any
import numpy as np


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the model
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode sentences into embeddings.

        Args:
            sentences: Single sentence or list of sentences to encode
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            convert_to_numpy: Whether to convert to numpy array
            normalize_embeddings: Whether to normalize embeddings
            **kwargs: Additional encoding parameters

        Returns:
            Embeddings as numpy array or list of arrays
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        pass

    @abstractmethod
    def get_max_seq_length(self) -> int:
        """Get the maximum sequence length supported."""
        pass

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config

    @abstractmethod
    def load_model(self) -> None:
        """Load the model weights."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        pass

    def warmup(self, sample_texts: Optional[List[str]] = None) -> None:
        """
        Warm up the model with sample texts.

        Args:
            sample_texts: Sample texts for warmup
        """
        if sample_texts is None:
            sample_texts = ["This is a warmup sentence."]

        _ = self.encode(sample_texts, show_progress_bar=False)

    def supports_matryoshka(self) -> bool:
        """Check if model supports Matryoshka representations."""
        return False

    def get_matryoshka_dimensions(self) -> Optional[List[int]]:
        """Get supported Matryoshka dimensions if available."""
        return None

    def encode_with_dimension(
        self, sentences: Union[str, List[str]], dimension: int, **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode with specific dimension (for Matryoshka models).

        Args:
            sentences: Sentences to encode
            dimension: Target dimension
            **kwargs: Additional parameters

        Returns:
            Embeddings with specified dimension
        """
        if not self.supports_matryoshka():
            raise NotImplementedError(
                f"Model {self.model_name} does not support Matryoshka representations"
            )

        embeddings = self.encode(sentences, **kwargs)

        # Truncate to specified dimension
        if isinstance(embeddings, np.ndarray):
            return embeddings[:, :dimension]
        else:
            return [emb[:dimension] for emb in embeddings]
