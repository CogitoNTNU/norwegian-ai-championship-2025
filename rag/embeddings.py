from langchain_ollama import OllamaEmbeddings

def get_embeddings_func():
    return OllamaEmbeddings(model="mxbai-embed-large")
