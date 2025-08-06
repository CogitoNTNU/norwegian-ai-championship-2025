import os
import json
import faiss
from sentence_transformers import SentenceTransformer


def create_and_save_faiss_index():
    """
    Loads documents from chunks.jsonl, generates BioBERT embeddings,
    and saves the FAISS index and document mappings.
    """
    print("Starting script to create FAISS index...")

    # Define paths
    script_dir = os.path.dirname(__file__)
    chunks_file_path = os.path.join(script_dir, "..", "chunking", "kg", "chunks.jsonl")
    output_dir = os.path.join(script_dir, "faiss_index")
    index_path = os.path.join(output_dir, "biobert_faiss.index")
    mapping_path = os.path.join(output_dir, "document_mapping.json")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load Documents ---
    print(f"Loading documents from {chunks_file_path}...")
    documents = []
    with open(chunks_file_path, "r", encoding="utf-8") as f:
        for line in f:
            documents.append(json.loads(line))

    if not documents:
        print("No documents found. Exiting.")
        return

    document_texts = [doc["text"] for doc in documents]
    print(f"Loaded {len(documents)} documents.")

    # --- 2. Generate Embeddings ---
    print("Loading BioBERT embedding model...")
    model = SentenceTransformer(
        "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    )

    print("Encoding documents... (This may take a while)")
    embeddings = model.encode(
        document_texts, show_progress_bar=True, convert_to_numpy=True
    )

    # Ensure embeddings are float32, as required by FAISS
    embeddings = embeddings.astype("float32")
    print(
        f"Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}."
    )

    # --- 3. Create and Populate FAISS Index ---
    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dimension)

    if faiss.get_num_gpus() > 0:
        print(f"Found {faiss.get_num_gpus()} GPUs. Using GPU for indexing.")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    print("Adding embeddings to the FAISS index...")
    index.add(embeddings)
    print(f"Successfully added {index.ntotal} vectors to the index.")

    # --- 4. Save Index and Mapping ---
    print(f"Saving FAISS index to {index_path}...")
    if faiss.get_num_gpus() > 0:
        # If on GPU, we need to move the index back to CPU before saving
        index_cpu = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index_cpu, index_path)
    else:
        faiss.write_index(index, index_path)

    # Save the document metadata for mapping results back
    print(f"Saving document mapping to {mapping_path}...")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=4)

    print("\n✨ FAISS index creation complete! ✨")
    print(f"Index file: {index_path}")
    print(f"Mapping file: {mapping_path}")


if __name__ == "__main__":
    create_and_save_faiss_index()
