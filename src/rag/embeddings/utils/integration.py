"""Integration utilities for updating existing DocumentStore."""

from pathlib import Path


def update_document_store():
    """
    Update the existing DocumentStore to use the new embedding system.

    This creates a new version that maintains backward compatibility.
    """

    # Path to the original document_store.py
    rag_pipeline_path = Path(__file__).parent.parent.parent / "rag-pipeline"

    # Create the updated DocumentStore
    updated_document_store = '''import json
from pathlib import Path
from typing import List, Dict, Optional
import faiss
from tqdm import tqdm
import sys

# Add src to path to import rag.embeddings
src_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(src_path))

from rag.embeddings.models import get_embedding_model
from rag.embeddings.managers import IndexManager


class DocumentStore:
    def __init__(
        self, 
        embedding_model: str = "all-MiniLM-L6-v2",
        use_new_embeddings: bool = True
    ):
        """
        Initialize document store with configurable embedding model.
        
        Args:
            embedding_model: Name of the embedding model to use
            use_new_embeddings: Whether to use new embedding system
        """
        self.use_new_embeddings = use_new_embeddings
        self.embedding_model_name = embedding_model
        
        if use_new_embeddings:
            # Use new embedding system
            self.embedding_model = get_embedding_model(embedding_model)
            self.index_manager = IndexManager()
        else:
            # Fallback to original implementation
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(embedding_model)
            
        self.index = None
        self.chunks = []
        self.chunk_metadata = []

    def load_medical_documents(self, topics_dir: str, topics_json: str) -> None:
        """Load and chunk medical documents from topics directory."""
        print("Loading medical documents...")

        # Load topic mapping
        with open(topics_json, "r") as f:
            topic_mapping = json.load(f)

        # Reverse mapping: topic_name -> topic_id
        name_to_id = {name: idx for name, idx in topic_mapping.items()}

        topics_path = Path(topics_dir)
        all_chunks = []
        all_metadata = []

        for topic_folder in tqdm(topics_path.iterdir()):
            if not topic_folder.is_dir():
                continue

            topic_name = topic_folder.name
            topic_id = name_to_id.get(topic_name, -1)

            # Look for markdown files in the topic folder
            for md_file in topic_folder.glob("*.md"):
                try:
                    with open(md_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Simple chunking: split by paragraphs
                    chunks = self._chunk_document(content)

                    for chunk in chunks:
                        if len(chunk.strip()) > 50:  # Only keep meaningful chunks
                            all_chunks.append(chunk)
                            all_metadata.append(
                                {
                                    "topic_name": topic_name,
                                    "topic_id": topic_id,
                                    "file": str(md_file),
                                }
                            )

                except Exception as e:
                    print(f"Error processing {md_file}: {e}")

        self.chunks = all_chunks
        self.chunk_metadata = all_metadata
        print(f"Loaded {len(self.chunks)} chunks from {len(name_to_id)} topics")

    def _chunk_document(self, content: str, chunk_size: int = 300) -> List[str]:
        """Simple paragraph-based chunking."""
        # Split by double newlines (paragraphs)
        paragraphs = content.split("\\n\\n")

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph would exceed chunk_size, start new chunk
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\\n\\n" + para if current_chunk else para

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def build_index(self) -> None:
        """Build FAISS index from document chunks."""
        if not self.chunks:
            raise ValueError("No documents loaded. Call load_medical_documents first.")

        print("Creating embeddings...")
        embeddings = self.embedding_model.encode(
            self.chunks, show_progress_bar=True, convert_to_numpy=True
        )

        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype("float32"))

        print(f"Index built with {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant document chunks."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")

        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding.astype("float32"), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):  # Valid index
                results.append(
                    {
                        "chunk": self.chunks[idx],
                        "metadata": self.chunk_metadata[idx],
                        "score": float(score),
                    }
                )

        return results

    def save_index(self, index_path: str) -> None:
        """Save the index and metadata to disk."""
        if self.use_new_embeddings and hasattr(self, 'index_manager'):
            # Use new index manager
            self.index_manager.save_index(
                self.index,
                self.embedding_model_name,
                self.chunks,
                self.chunk_metadata
            )
        else:
            # Original implementation
            if self.index is None:
                raise ValueError("No index to save")

            faiss.write_index(self.index, f"{index_path}.faiss")

            # Save chunks and metadata
            data = {"chunks": self.chunks, "chunk_metadata": self.chunk_metadata}
            with open(f"{index_path}.json", "w") as f:
                json.dump(data, f)

        print(f"Index saved to {index_path}")

    def load_index(self, index_path: str) -> None:
        """Load the index and metadata from disk."""
        if self.use_new_embeddings and hasattr(self, 'index_manager'):
            # Try to load with index manager
            if self.index_manager.index_exists(self.embedding_model_name):
                self.index, self.chunks, self.chunk_metadata = (
                    self.index_manager.load_index(self.embedding_model_name)
                )
                print(f"Index loaded for model {self.embedding_model_name}")
                return
        
        # Fallback to original implementation
        self.index = faiss.read_index(f"{index_path}.faiss")

        with open(f"{index_path}.json", "r") as f:
            data = json.load(f)

        self.chunks = data["chunks"]
        self.chunk_metadata = data["chunk_metadata"]

        print(f"Index loaded from {index_path}")
'''

    # Save the updated DocumentStore
    output_path = rag_pipeline_path / "document_store_updated.py"
    with open(output_path, "w") as f:
        f.write(updated_document_store)

    print(f"Updated DocumentStore saved to: {output_path}")
    print("\nTo use the new embedding system:")
    print("1. Backup the original: cp document_store.py document_store_original.py")
    print("2. Replace with updated: cp document_store_updated.py document_store.py")
    print("\nOr import directly in your code:")
    print("from document_store_updated import DocumentStore")
