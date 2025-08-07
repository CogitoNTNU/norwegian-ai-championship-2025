from langchain_ollama import OllamaEmbeddings

def get_embeddings_func():
    return OllamaEmbeddings(model="snowflake-arctic-embed2:latest")
