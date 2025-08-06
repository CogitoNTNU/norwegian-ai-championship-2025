#!/usr/bin/env python3
"""
Create topic-specific BM25 indexes to avoid rebuilding them during evaluation.
This pre-computes a BM25 index for each topic.
"""

import json
import pickle
import bm25s
import Stemmer
import re
from pathlib import Path
from collections import defaultdict


def create_topic_specific_indexes():
    """
    Creates and saves BM25 indexes for each topic separately.
    """
    print("🚀 Creating topic-specific BM25 indexes...")

    # Define paths
    script_dir = Path(__file__).parent
    chunks_file_path = script_dir / ".." / "chunking" / "kg" / "chunks.jsonl"
    output_dir = script_dir / "optimized_indexes" / "topic_bm25"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Documents ---
    print(f"📄 Loading documents from {chunks_file_path}...")
    documents = []
    with open(chunks_file_path, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))

    if not documents:
        print("❌ No documents found. Exiting.")
        return

    # --- 2. Group by Topic ---
    print("📊 Grouping documents by topic...")
    topic_docs = defaultdict(list)
    for doc in documents:
        topic_name = doc.get("topic_name", "Unknown")
        topic_docs[topic_name].append(doc)

    print(f"✅ Found {len(topic_docs)} topics:")
    for topic, docs in topic_docs.items():
        print(f"   • {topic}: {len(docs)} documents")

    # --- 3. Create Topic-Specific BM25 Indexes ---
    stemmer = Stemmer.Stemmer("english")

    def tokenize_and_stem(text: str):
        """Tokenize and stem text for BM25 processing."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        return stemmer.stemWords(tokens)

    topic_index_mapping = {}

    for topic_name, docs in topic_docs.items():
        print(f"🔤 Creating BM25 index for topic: {topic_name}")

        # Extract texts and create mapping
        topic_texts = [doc["text"] for doc in docs]
        tokenized_corpus = [tokenize_and_stem(text) for text in topic_texts]

        # Create BM25 index
        bm25_index = bm25s.BM25()
        bm25_index.index(tokenized_corpus)

        # Save index
        safe_topic_name = re.sub(r"[^\w\s-]", "", topic_name).strip().replace(" ", "_")
        index_path = output_dir / f"bm25_{safe_topic_name}.pkl"
        mapping_path = output_dir / f"mapping_{safe_topic_name}.json"

        with open(index_path, "wb") as f:
            pickle.dump(bm25_index, f)

        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2)

        topic_index_mapping[topic_name] = {
            "index_file": str(index_path.name),
            "mapping_file": str(mapping_path.name),
            "document_count": len(docs),
        }

        print(f"   ✅ Saved {len(docs)} documents to {index_path.name}")

    # --- 4. Save Topic Index Mapping ---
    mapping_file = output_dir / "topic_index_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(topic_index_mapping, f, indent=2)

    print("\n✨ Topic-specific indexes created successfully! ✨")
    print(f"📁 Output directory: {output_dir}")
    print(f"📋 Index mapping: {mapping_file}")
    print("\n📈 Statistics:")
    print(f"   • Total topics: {len(topic_docs)}")
    print(f"   • Total documents: {len(documents)}")


if __name__ == "__main__":
    create_topic_specific_indexes()
