"""Index manager for managing FAISS indices with different embedding models."""

from pathlib import Path
from typing import Dict, Optional, Any, List
import json
import faiss
from datetime import datetime


class IndexManager:
    """Manages FAISS indices for different embedding models."""

    def __init__(self, index_dir: Optional[str] = None):
        """
        Initialize index manager.

        Args:
            index_dir: Directory for storing indices
        """
        self.index_dir = Path(index_dir or "indices")
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Metadata about indices
        self.metadata_file = self.index_dir / "index_metadata.json"
        self.metadata = self._load_metadata()

        # Cache loaded indices
        self.loaded_indices: Dict[str, faiss.Index] = {}

    def _load_metadata(self) -> Dict[str, Any]:
        """Load index metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self) -> None:
        """Save index metadata."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def create_index(
        self,
        model_name: str,
        dimension: int,
        index_type: str = "Flat",
        metric: str = "cosine",
    ) -> faiss.Index:
        """
        Create a new FAISS index.

        Args:
            model_name: Name of the embedding model
            dimension: Embedding dimension
            index_type: Type of index (Flat, IVF, HNSW)
            metric: Distance metric (cosine, l2)

        Returns:
            FAISS index
        """
        # Select metric
        if metric == "cosine":
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            metric_type = faiss.METRIC_L2

        # Create index based on type
        if index_type == "Flat":
            index = (
                faiss.IndexFlatIP(dimension)
                if metric == "cosine"
                else faiss.IndexFlatL2(dimension)
            )
        elif index_type == "IVF":
            # IVF index for larger datasets
            quantizer = (
                faiss.IndexFlatIP(dimension)
                if metric == "cosine"
                else faiss.IndexFlatL2(dimension)
            )
            index = faiss.IndexIVFFlat(quantizer, dimension, 100, metric_type)
        elif index_type == "HNSW":
            # HNSW for fast approximate search
            index = faiss.IndexHNSWFlat(dimension, 32, metric_type)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Store metadata
        self.metadata[model_name] = {
            "dimension": dimension,
            "index_type": index_type,
            "metric": metric,
            "created": datetime.now().isoformat(),
            "num_vectors": 0,
        }
        self._save_metadata()

        return index

    def save_index(
        self,
        index: faiss.Index,
        model_name: str,
        chunks: List[str],
        chunk_metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Save an index to disk.

        Args:
            index: FAISS index
            model_name: Model name
            chunks: Text chunks
            chunk_metadata: Metadata for chunks
        """
        # Create model directory
        model_dir = self.index_dir / model_name.replace("/", "_")
        model_dir.mkdir(exist_ok=True)

        # Save FAISS index
        index_file = model_dir / "index.faiss"
        faiss.write_index(index, str(index_file))

        # Save chunks and metadata
        data = {"chunks": chunks, "chunk_metadata": chunk_metadata}
        data_file = model_dir / "data.json"
        with open(data_file, "w") as f:
            json.dump(data, f)

        # Update metadata
        if model_name in self.metadata:
            self.metadata[model_name]["num_vectors"] = index.ntotal
            self.metadata[model_name]["last_updated"] = datetime.now().isoformat()
        self._save_metadata()

    def load_index(
        self, model_name: str
    ) -> tuple[faiss.Index, List[str], List[Dict[str, Any]]]:
        """
        Load an index from disk.

        Args:
            model_name: Model name

        Returns:
            Tuple of (index, chunks, chunk_metadata)
        """
        # Check cache first
        if model_name in self.loaded_indices:
            index = self.loaded_indices[model_name]
        else:
            # Load from disk
            model_dir = self.index_dir / model_name.replace("/", "_")
            index_file = model_dir / "index.faiss"

            if not index_file.exists():
                raise ValueError(f"No index found for model {model_name}")

            index = faiss.read_index(str(index_file))
            self.loaded_indices[model_name] = index

        # Load data
        model_dir = self.index_dir / model_name.replace("/", "_")
        data_file = model_dir / "data.json"

        with open(data_file, "r") as f:
            data = json.load(f)

        return index, data["chunks"], data["chunk_metadata"]

    def index_exists(self, model_name: str) -> bool:
        """Check if an index exists for a model."""
        model_dir = self.index_dir / model_name.replace("/", "_")
        return (model_dir / "index.faiss").exists()

    def get_index_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about an index."""
        if model_name not in self.metadata:
            return {"exists": False}

        info = self.metadata[model_name].copy()
        info["exists"] = self.index_exists(model_name)

        # Add size information if index exists
        if info["exists"]:
            model_dir = self.index_dir / model_name.replace("/", "_")
            index_file = model_dir / "index.faiss"
            info["size_mb"] = index_file.stat().st_size / (1024 * 1024)

        return info

    def list_indices(self) -> List[str]:
        """List all available indices."""
        return [name for name in self.metadata.keys() if self.index_exists(name)]

    def delete_index(self, model_name: str) -> None:
        """Delete an index."""
        model_dir = self.index_dir / model_name.replace("/", "_")
        if model_dir.exists():
            import shutil

            shutil.rmtree(model_dir)

        if model_name in self.metadata:
            del self.metadata[model_name]
            self._save_metadata()

        if model_name in self.loaded_indices:
            del self.loaded_indices[model_name]

    def compare_indices(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare multiple indices."""
        comparison = {}

        for model_name in model_names:
            if model_name in self.metadata:
                info = self.get_index_info(model_name)
                comparison[model_name] = {
                    "dimension": info.get("dimension"),
                    "num_vectors": info.get("num_vectors"),
                    "size_mb": info.get("size_mb", 0),
                    "index_type": info.get("index_type"),
                    "metric": info.get("metric"),
                    "last_updated": info.get("last_updated"),
                }

        return comparison
