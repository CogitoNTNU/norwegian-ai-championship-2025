"""Document manager for shared chunk storage across different embedding models."""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm


class DocumentManager:
    """Manages document chunks shared across different embedding models."""
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize document manager.
        
        Args:
            storage_dir: Directory for storing documents
        """
        self.storage_dir = Path(storage_dir or "indices")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.chunks_file = self.storage_dir / "documents.json"
        self.chunks_hash_file = self.storage_dir / "documents.hash"
        
        # Cache
        self.chunks: Optional[List[str]] = None
        self.chunk_metadata: Optional[List[Dict[str, Any]]] = None
        self.topics_mapping: Optional[Dict[str, int]] = None
    
    def _compute_documents_hash(self, topics_dir: Path) -> str:
        """Compute hash of all documents to detect changes."""
        hasher = hashlib.md5()
        
        # Hash all markdown files in topics directory
        for md_file in sorted(topics_dir.rglob("*.md")):
            hasher.update(str(md_file).encode())
            hasher.update(str(md_file.stat().st_mtime).encode())
        
        return hasher.hexdigest()
    
    def _load_topics_mapping(self, topics_json: Path) -> Dict[str, int]:
        """Load topic name to ID mapping."""
        with open(topics_json, "r") as f:
            return json.load(f)
    
    def _chunk_document(self, content: str, chunk_size: int = 400) -> List[str]:
        """
        Chunk document into smaller pieces.
        
        Args:
            content: Document content
            chunk_size: Maximum chunk size in words
            
        Returns:
            List of chunks
        """
        # Split by paragraphs first
        paragraphs = content.split("\n\n")
        
        chunks = []
        current_chunk = ""
        current_words = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_words = len(para.split())
            
            # If paragraph itself is too long, split by sentences
            if para_words > chunk_size:
                sentences = para.split(". ")
                for sent in sentences:
                    sent_words = len(sent.split())
                    if current_words + sent_words > chunk_size and current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = sent + ". "
                        current_words = sent_words
                    else:
                        current_chunk += sent + ". "
                        current_words += sent_words
            else:
                # If adding paragraph exceeds chunk size, start new chunk
                if current_words + para_words > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = para
                    current_words = para_words
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
                    current_words += para_words
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def load_or_build_chunks(
        self, 
        topics_dir: str, 
        topics_json: str,
        force_rebuild: bool = False
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Load existing chunks or build new ones from documents.
        
        Args:
            topics_dir: Path to topics directory
            topics_json: Path to topics JSON mapping
            force_rebuild: Force rebuilding chunks
            
        Returns:
            Tuple of (chunks, chunk_metadata)
        """
        topics_path = Path(topics_dir)
        topics_json_path = Path(topics_json)
        
        # Load topics mapping
        self.topics_mapping = self._load_topics_mapping(topics_json_path)
        
        # Check if we need to rebuild
        current_hash = self._compute_documents_hash(topics_path)
        
        if not force_rebuild and self.chunks_file.exists() and self.chunks_hash_file.exists():
            # Check if documents haven't changed
            stored_hash = self.chunks_hash_file.read_text().strip()
            if stored_hash == current_hash:
                # Load cached chunks
                print("Loading cached document chunks...")
                with open(self.chunks_file, "r") as f:
                    data = json.load(f)
                self.chunks = data["chunks"]
                self.chunk_metadata = data["metadata"]
                print(f"Loaded {len(self.chunks)} chunks from cache")
                return self.chunks, self.chunk_metadata
        
        # Build new chunks
        print("Building document chunks...")
        self.chunks = []
        self.chunk_metadata = []
        
        # Process each topic folder
        for topic_folder in tqdm(list(topics_path.iterdir())):
            if not topic_folder.is_dir():
                continue
            
            topic_name = topic_folder.name
            topic_id = self.topics_mapping.get(topic_name, -1)
            
            # Process all markdown files in topic
            for md_file in topic_folder.glob("*.md"):
                try:
                    with open(md_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    # Chunk the document
                    doc_chunks = self._chunk_document(content)
                    
                    # Add chunks with metadata
                    for chunk_idx, chunk in enumerate(doc_chunks):
                        if len(chunk.strip()) > 50:  # Skip very short chunks
                            self.chunks.append(chunk)
                            self.chunk_metadata.append({
                                "topic_name": topic_name,
                                "topic_id": topic_id,
                                "file": str(md_file.relative_to(topics_path)),
                                "chunk_index": chunk_idx
                            })
                
                except Exception as e:
                    print(f"Error processing {md_file}: {e}")
        
        # Save chunks and hash
        print(f"Saving {len(self.chunks)} chunks to cache...")
        with open(self.chunks_file, "w") as f:
            json.dump({
                "chunks": self.chunks,
                "metadata": self.chunk_metadata
            }, f, indent=2)
        
        self.chunks_hash_file.write_text(current_hash)
        
        return self.chunks, self.chunk_metadata
    
    def get_chunks(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Get loaded chunks and metadata."""
        if self.chunks is None:
            raise ValueError("No chunks loaded. Call load_or_build_chunks first.")
        return self.chunks, self.chunk_metadata
    
    def get_chunk_by_indices(self, indices: List[int]) -> List[Dict[str, Any]]:
        """Get specific chunks by their indices."""
        if self.chunks is None:
            raise ValueError("No chunks loaded.")
        
        results = []
        for idx in indices:
            if 0 <= idx < len(self.chunks):
                results.append({
                    "chunk": self.chunks[idx],
                    "metadata": self.chunk_metadata[idx]
                })
        
        return results