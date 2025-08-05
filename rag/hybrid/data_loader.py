#!/usr/bin/env python3
"""
Phase 2: Document Chunk Data Loader
Loads rich document chunks with metadata for improved RAG performance.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from . import config


class DocumentChunkLoader:
    """Loads document chunks and metadata for Phase 2 RAG system."""
    
    def __init__(self):
        self.chunks_dir = config.DOCUMENT_CHUNKS_DIR
        self.metadata_file = config.CHUNK_METADATA_FILE
        self.topic_mapping_file = config.TOPIC_MAPPING_FILE
        
        # Cached data
        self._chunk_metadata = None
        self._topic_mapping = None
    
    def load_chunk_metadata(self) -> List[Dict]:
        """Load chunk metadata from JSON file."""
        if self._chunk_metadata is None:
            if not self.metadata_file.exists():
                raise FileNotFoundError(f"Chunk metadata file not found: {self.metadata_file}")
            
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self._chunk_metadata = json.load(f)
        
        return self._chunk_metadata
    
    def load_topic_mapping(self) -> Dict[str, int]:
        """Load topic mapping from JSON file."""
        if self._topic_mapping is None:
            if not self.topic_mapping_file.exists():
                raise FileNotFoundError(f"Topic mapping file not found: {self.topic_mapping_file}")
            
            with open(self.topic_mapping_file, 'r', encoding='utf-8') as f:
                self._topic_mapping = json.load(f)
        
        return self._topic_mapping
    
    def load_chunk_content(self, chunk_id: str) -> str:
        """Load content of a specific chunk."""
        chunk_file = self.chunks_dir / f"{chunk_id}.txt"
        
        if not chunk_file.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_file}")
        
        with open(chunk_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def load_all_chunks(self) -> Tuple[List[str], List[Dict]]:
        """Load all chunk contents and their metadata."""
        metadata = self.load_chunk_metadata()
        
        chunks = []
        chunk_metadata = []
        
        for meta in metadata:
            try:
                content = self.load_chunk_content(meta['chunk_id'])
                chunks.append(content)
                chunk_metadata.append(meta)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
        
        print(f"Loaded {len(chunks)} document chunks")
        return chunks, chunk_metadata
    
    def get_chunks_by_topic(self, topic_id: int) -> Tuple[List[str], List[Dict]]:
        """Get all chunks for a specific topic."""
        all_chunks, all_metadata = self.load_all_chunks()
        
        topic_chunks = []
        topic_metadata = []
        
        for chunk, meta in zip(all_chunks, all_metadata):
            if meta['topic_id'] == topic_id:
                topic_chunks.append(chunk)
                topic_metadata.append(meta)
        
        return topic_chunks, topic_metadata
    
    def get_document_info(self, chunk_id: str) -> Optional[Dict]:
        """Get metadata for a specific chunk."""
        metadata = self.load_chunk_metadata()
        
        for meta in metadata:
            if meta['chunk_id'] == chunk_id:
                return meta
        
        return None


class StatementCompatibilityLoader:
    """
    Backward compatibility loader to maintain interface with statement-based system.
    Maps document chunks to the expected statement format.
    """
    
    def __init__(self):
        self.chunk_loader = DocumentChunkLoader()
    
    def load_data(self) -> Tuple[List[str], List[Dict]]:
        """Load document chunks formatted as statements for compatibility."""
        chunks, metadata = self.chunk_loader.load_all_chunks()
        
        # Convert chunks to statement-like format
        statements = []
        statement_metadata = []
        
        for chunk, meta in zip(chunks, metadata):
            # Create statement-like record
            statement_record = {
                "id": meta['chunk_id'],
                "text": chunk,
                "topic_id": meta['topic_id'],
                "topic_name": meta['topic_name'],
                "source": meta['source_document'],
                # Additional rich context metadata
                "chunk_index": meta['chunk_index'],
                "word_count": meta['word_count']
            }
            
            statements.append(chunk)
            statement_metadata.append(statement_record)
        
        return statements, statement_metadata


def get_data_loader() -> StatementCompatibilityLoader:
    """Get the appropriate data loader for the current configuration."""
    return StatementCompatibilityLoader()


# Legacy function for backward compatibility
def load_statements() -> Tuple[List[str], List[Dict]]:
    """Legacy function to load statements - now loads document chunks."""
    loader = get_data_loader()
    return loader.load_data()
