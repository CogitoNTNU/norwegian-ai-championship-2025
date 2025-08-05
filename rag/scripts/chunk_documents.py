from pathlib import Path

# Directory where the rich document chunks will be stored
output_dir = Path(
    "/Users/nybruker/Documents/nm-ai/norwegian-ai-championship-2025/rag/data/processed/chunks"
)
output_dir.mkdir(parents=True, exist_ok=True)


# Function to read files into chunks
def read_and_chunkify(file_path, chunk_size=400):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        words = content.split()
        # Chunk the document into parts of approximately `chunk_size` words
        return [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]


# Function to process all markdown files and produce chunks
def process_documents(raw_docs_path):
    chunk_id = 0
    for filepath in raw_docs_path.rglob("*.md"):
        chunks = read_and_chunkify(filepath)
        base_name = filepath.stem
        for i, chunk in enumerate(chunks):
            chunk_file = output_dir / f"{base_name}_chunk_{i}.txt"
            with open(chunk_file, "w", encoding="utf-8") as chunk_f:
                chunk_f.write(chunk)
            chunk_id += 1


# Call the function
process_documents(
    Path(
        "/Users/nybruker/Documents/nm-ai/norwegian-ai-championship-2025/rag/data/raw/topics"
    )
)
print("Document processing to chunks completed.")
