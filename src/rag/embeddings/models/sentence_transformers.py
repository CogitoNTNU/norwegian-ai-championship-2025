"""Sentence Transformers wrapper for embedding models."""

from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from .base import BaseEmbeddingModel


class SentenceTransformerModel(BaseEmbeddingModel):
    """Wrapper for sentence-transformers models."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        matryoshka_dims: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Initialize sentence transformer model.

        Args:
            model_name: Name of the model from HuggingFace
            device: Device to use (cuda/cpu)
            cache_folder: Folder to cache models
            matryoshka_dims: Supported Matryoshka dimensions
            **kwargs: Additional parameters
        """
        super().__init__(model_name, **kwargs)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_folder = cache_folder
        self.matryoshka_dims = matryoshka_dims
        self.model = None

        # Load model on initialization
        self.load_model()

    def load_model(self) -> None:
        """Load the sentence transformer model."""
        if self.model is None:
            self.model = SentenceTransformer(
                self.model_name, device=self.device, cache_folder=self.cache_folder
            )

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()

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
            sentences: Single sentence or list of sentences
            batch_size: Batch size for encoding
            show_progress_bar: Show progress bar
            convert_to_numpy: Convert to numpy array
            normalize_embeddings: Normalize embeddings
            **kwargs: Additional parameters

        Returns:
            Embeddings array
        """
        if self.model is None:
            self.load_model()

        return self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
            **kwargs,
        )

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self.model is None:
            self.load_model()
        return self.model.get_sentence_embedding_dimension()

    def get_max_seq_length(self) -> int:
        """Get maximum sequence length."""
        if self.model is None:
            self.load_model()
        return self.model.max_seq_length

    def supports_matryoshka(self) -> bool:
        """Check if model supports Matryoshka."""
        return self.matryoshka_dims is not None

    def get_matryoshka_dimensions(self) -> Optional[List[int]]:
        """Get Matryoshka dimensions."""
        return self.matryoshka_dims


class MatryoshkaModel(SentenceTransformerModel):
    """Specialized model for Matryoshka representations."""

    def __init__(self, model_name: str, matryoshka_dims: List[int], **kwargs):
        """
        Initialize Matryoshka model.

        Args:
            model_name: Model name
            matryoshka_dims: List of supported dimensions
            **kwargs: Additional parameters
        """
        super().__init__(model_name, matryoshka_dims=matryoshka_dims, **kwargs)

    def encode_with_dimension(
        self, sentences: Union[str, List[str]], dimension: int, **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode with specific dimension.

        Args:
            sentences: Sentences to encode
            dimension: Target dimension
            **kwargs: Additional parameters

        Returns:
            Truncated embeddings
        """
        if dimension not in self.matryoshka_dims:
            raise ValueError(
                f"Dimension {dimension} not supported. "
                f"Supported dimensions: {self.matryoshka_dims}"
            )

        # Get full embeddings
        embeddings = self.encode(sentences, **kwargs)

        # Truncate to specified dimension
        if isinstance(embeddings, np.ndarray):
            return embeddings[:, :dimension]
        else:
            return [emb[:dimension] for emb in embeddings]


class MedicalEmbeddingModel(SentenceTransformerModel):
    """Specialized wrapper for medical embedding models."""

    def __init__(
        self, model_name: str, preprocessing_fn: Optional[callable] = None, **kwargs
    ):
        """
        Initialize medical embedding model.

        Args:
            model_name: Model name
            preprocessing_fn: Optional preprocessing function
            **kwargs: Additional parameters
        """
        super().__init__(model_name, **kwargs)
        self.preprocessing_fn = preprocessing_fn

    def encode(
        self, sentences: Union[str, List[str]], **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode with optional preprocessing.

        Args:
            sentences: Sentences to encode
            **kwargs: Additional parameters

        Returns:
            Embeddings
        """
        # Apply preprocessing if available
        if self.preprocessing_fn is not None:
            if isinstance(sentences, str):
                sentences = self.preprocessing_fn(sentences)
            else:
                sentences = [self.preprocessing_fn(s) for s in sentences]

        return super().encode(sentences, **kwargs)
