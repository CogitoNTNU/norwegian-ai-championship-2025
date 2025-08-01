import os
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class DocumentStore:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        
    def load_medical_documents(self, topics_dir: str, topics_json: str) -> None:
        """Load and chunk medical documents from topics directory."""
        print("Loading medical documents...")
        
        # Load topic mapping
        with open(topics_json, 'r') as f:
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
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Simple chunking: split by paragraphs
                    chunks = self._chunk_document(content)
                    
                    for chunk in chunks:
                        if len(chunk.strip()) > 50:  # Only keep meaningful chunks
                            all_chunks.append(chunk)
                            all_metadata.append({
                                'topic_name': topic_name,
                                'topic_id': topic_id,
                                'file': str(md_file)
                            })
                            
                except Exception as e:
                    print(f"Error processing {md_file}: {e}")
        
        self.chunks = all_chunks
        self.chunk_metadata = all_metadata
        print(f"Loaded {len(self.chunks)} chunks from {len(name_to_id)} topics")
    
    def _chunk_document(self, content: str, chunk_size: int = 300) -> List[str]:
        """Simple paragraph-based chunking."""
        # Split by double newlines (paragraphs)
        paragraphs = content.split('\n\n')
        
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
                current_chunk += "\n\n" + para if current_chunk else para
        
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
            self.chunks, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant document chunks."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):  # Valid index
                results.append({
                    'chunk': self.chunks[idx],
                    'metadata': self.chunk_metadata[idx],
                    'score': float(score)
                })
        
        return results
    
    def save_index(self, index_path: str) -> None:
        """Save the index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        
        faiss.write_index(self.index, f"{index_path}.faiss")
        
        # Save chunks and metadata
        data = {
            'chunks': self.chunks,
            'chunk_metadata': self.chunk_metadata
        }
        with open(f"{index_path}.json", 'w') as f:
            json.dump(data, f)
        
        print(f"Index saved to {index_path}")
    
    def load_index(self, index_path: str) -> None:
        """Load the index and metadata from disk."""
        self.index = faiss.read_index(f"{index_path}.faiss")
        
        with open(f"{index_path}.json", 'r') as f:
            data = json.load(f)
        
        self.chunks = data['chunks']
        self.chunk_metadata = data['chunk_metadata']
        
        print(f"Index loaded from {index_path}")
