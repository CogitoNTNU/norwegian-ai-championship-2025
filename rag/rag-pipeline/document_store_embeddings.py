"""Document store using configurable embedding models."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from tqdm import tqdm

# Add parent directory to path for embeddings import
rag_path = Path(__file__).parent.parent
sys.path.insert(0, str(rag_path))

from embeddings.managers import DocumentManager, IndexManager  # noqa: E402
from embeddings.models import get_embedding_model  # noqa: E402


class EmbeddingsDocumentStore:
    """Document store with configurable embedding models."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize document store with specified embedding model.

        Args:
            embedding_model: Name of embedding model from registry
        """
        self.embedding_model_name = embedding_model
        self.embedding_model = get_embedding_model(embedding_model)
        # Use rag/indices directory
        indices_dir = str(rag_path / "indices")
        self.index_manager = IndexManager(index_dir=indices_dir)
        self.document_manager = DocumentManager(storage_dir=indices_dir)

        # Index and metadata
        self.index = None
        self.chunks = []
        self.chunk_metadata = []

        print(f"Initialized document store with model: {embedding_model}")

    def load_medical_documents(self, topics_dir: str, topics_json: str) -> None:
        """
        Load and chunk medical documents from topics directory.

        Args:
            topics_dir: Path to topics directory
            topics_json: Path to topics mapping JSON
        """
        # Use document manager to load or build chunks
        self.chunks, self.chunk_metadata = self.document_manager.load_or_build_chunks(
            topics_dir, topics_json
        )

        print(f"Loaded {len(self.chunks)} document chunks")

    def build_index(self) -> None:
        """Build FAISS index from document chunks."""
        if not self.chunks:
            raise ValueError("No documents loaded. Call load_medical_documents first.")

        # Check if index already exists for this model
        if self.index_manager.index_exists(self.embedding_model_name):
            print(f"Loading existing index for {self.embedding_model_name}...")
            self.index, _, _ = self.index_manager.load_index(self.embedding_model_name)
            return

        print(f"Creating embeddings with {self.embedding_model_name}...")

        # Process in batches for memory efficiency
        # Use smaller batch size for large models
        batch_size = 8 if "stella" in self.embedding_model_name.lower() else 32
        all_embeddings = []

        # Process in small, safe batches
        safe_batch_size = 5  # Very small batch size for safety
        print(f"Total chunks to process: {len(self.chunks)}")

        for i in tqdm(range(0, len(self.chunks), safe_batch_size)):
            batch = self.chunks[i : i + safe_batch_size]
            try:
                # Use the wrapper's encode method to get macOS safety features
                batch_embeddings = self.embedding_model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=safe_batch_size,
                )

                # Ensure it's 2D
                if len(batch_embeddings.shape) == 1:
                    batch_embeddings = batch_embeddings.reshape(1, -1)

                all_embeddings.append(batch_embeddings)

            except Exception as e:
                print(f"\nError encoding batch starting at {i}: {e}")
                print(f"Batch size: {len(batch)}")
                print(f"First chunk preview: {batch[0][:100]}...")
                raise

        # Concatenate all embeddings
        embeddings = np.vstack(all_embeddings)

        print("Building FAISS index...")
        dimension = embeddings.shape[1]

        # Create appropriate index based on dataset size
        if len(self.chunks) < 10000:
            # Use flat index for small datasets (exact search)
            self.index = self.index_manager.create_index(
                self.embedding_model_name, dimension, index_type="Flat", metric="cosine"
            )
        else:
            # Use IVF for larger datasets (approximate search)
            self.index = self.index_manager.create_index(
                self.embedding_model_name, dimension, index_type="IVF", metric="cosine"
            )
            # Train IVF index
            self.index.train(embeddings.astype("float32"))

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add embeddings to index
        self.index.add(embeddings.astype("float32"))

        # Save index
        self.index_manager.save_index(
            self.index, self.embedding_model_name, self.chunks, self.chunk_metadata
        )

        print(f"Index built with {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of results with chunks, metadata, and scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")

        # Encode query
        query_embedding = self.embedding_model.encode(
            [query], convert_to_numpy=True, show_progress_bar=False
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding.astype("float32"), k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):  # Valid index
                results.append(
                    {
                        "chunk": self.chunks[idx],
                        "metadata": self.chunk_metadata[idx],
                        "score": float(score),
                    }
                )

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        return {
            "model_name": self.embedding_model_name,
            "supports_matryoshka": self.embedding_model.supports_matryoshka(),
            "dimension": self.embedding_model.get_dimension(),
            "index_size": self.index.ntotal if self.index else 0,
            "num_chunks": len(self.chunks),
        }

    def save_index(self, index_path: str) -> None:
        """Save the index to disk (for compatibility)."""
        if self.index is None:
            raise ValueError("No index to save")

        # Use index manager to save
        self.index_manager.save_index(
            self.index, self.embedding_model_name, self.chunks, self.chunk_metadata
        )

        print(f"Index saved for model {self.embedding_model_name}")

    def load_index(self, index_path: str) -> None:
        """Load the index from disk (for compatibility)."""
        # Try to load from index manager
        if self.index_manager.index_exists(self.embedding_model_name):
            self.index, self.chunks, self.chunk_metadata = (
                self.index_manager.load_index(self.embedding_model_name)
            )
            print(f"Index loaded for model {self.embedding_model_name}")
        else:
            raise ValueError(f"No index found for model {self.embedding_model_name}")

