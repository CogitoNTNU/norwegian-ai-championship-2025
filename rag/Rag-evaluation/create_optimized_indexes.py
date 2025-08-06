#!/usr/bin/env python3
"""
Create optimized FAISS and BM25 indexes from the new chunking data.
This will pre-compute all embeddings and tokenization for faster retrieval.
"""

import os
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import bm25s
import Stemmer
import re
from pathlib import Path

def create_optimized_indexes():
    """
    Creates and saves both FAISS (semantic) and BM25 (keyword) indexes
    from the improved chunking data.
    """
    print("🚀 Creating optimized FAISS and BM25 indexes...")

    # Define paths
    script_dir = Path(__file__).parent
    chunks_file_path = script_dir / ".." / "chunking" / "kg" / "chunks.jsonl"
    output_dir = script_dir / "optimized_indexes"
    
    # Output files
    faiss_index_path = output_dir / "biobert_faiss.index"
    bm25_index_path = output_dir / "bm25_index.pkl"
    mapping_path = output_dir / "document_mapping.json"
    metadata_path = output_dir / "index_metadata.json"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # --- 1. Load Documents ---
    print(f"📄 Loading documents from {chunks_file_path}...")
    documents = []
    with open(chunks_file_path, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))
    
    if not documents:
        print("❌ No documents found. Exiting.")
        return

    document_texts = [doc['text'] for doc in documents]
    print(f"✅ Loaded {len(documents)} documents from {len(set(doc['topic_name'] for doc in documents))} topics.")

    # --- 2. Create FAISS Index ---
    print("🧠 Loading BioBERT embedding model...")
    model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    
    print("🔄 Encoding documents for semantic search... (This may take a while)")
    embeddings = model.encode(document_texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype('float32')
    print(f"✅ Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}.")

    # Create FAISS index
    embedding_dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(embedding_dimension)
    
    # Use GPU if available
    if faiss.get_num_gpus() > 0:
        print(f"🎮 Found {faiss.get_num_gpus()} GPUs. Using GPU for indexing.")
        res = faiss.StandardGpuResources()
        faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)

    print("🔍 Adding embeddings to FAISS index...")
    faiss_index.add(embeddings)
    print(f"✅ FAISS index created with {faiss_index.ntotal} vectors.")

    # --- 3. Create BM25 Index ---
    print("🔤 Creating BM25 index for keyword search...")
    stemmer = Stemmer.Stemmer("english")
    
    def tokenize_and_stem(text: str):
        """Tokenize and stem text for BM25 processing."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        return stemmer.stemWords(tokens)
    
    print("⚡ Tokenizing and stemming documents...")
    tokenized_corpus = [tokenize_and_stem(doc) for doc in document_texts]
    
    bm25_index = bm25s.BM25()
    bm25_index.index(tokenized_corpus)
    print(f"✅ BM25 index created for {len(tokenized_corpus)} documents.")

    # --- 4. Save Indexes ---
    print("💾 Saving indexes...")
    
    # Save FAISS index
    if faiss.get_num_gpus() > 0:
        # Move back to CPU for saving
        faiss_index_cpu = faiss.index_gpu_to_cpu(faiss_index)
        faiss.write_index(faiss_index_cpu, str(faiss_index_path))
    else:
        faiss.write_index(faiss_index, str(faiss_index_path))
    
    # Save BM25 index
    with open(bm25_index_path, "wb") as f:
        pickle.dump(bm25_index, f)
    
    # Save document mapping
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)
    
    # Save metadata
    metadata = {
        "total_documents": len(documents),
        "embedding_dimension": embedding_dimension,
        "unique_topics": len(set(doc['topic_name'] for doc in documents)),
        "unique_articles": len(set(doc['article_title'] for doc in documents)),
        "avg_chunk_words": sum(doc['word_count'] for doc in documents) / len(documents),
        "model_name": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        "created_from": str(chunks_file_path),
        "files": {
            "faiss_index": str(faiss_index_path.name),
            "bm25_index": str(bm25_index_path.name),
            "document_mapping": str(mapping_path.name),
        }
    }
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✨ Optimized indexes created successfully! ✨")
    print(f"📁 Output directory: {output_dir}")
    print(f"🧠 FAISS index: {faiss_index_path}")  
    print(f"🔤 BM25 index: {bm25_index_path}")
    print(f"📋 Document mapping: {mapping_path}")
    print(f"📊 Metadata: {metadata_path}")
    print(f"\n📈 Statistics:")
    print(f"   • Total documents: {metadata['total_documents']:,}")
    print(f"   • Unique topics: {metadata['unique_topics']}")
    print(f"   • Unique articles: {metadata['unique_articles']}")
    print(f"   • Average chunk size: {metadata['avg_chunk_words']:.1f} words")

if __name__ == "__main__":
    create_optimized_indexes()
