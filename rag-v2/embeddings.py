from langchain_ollama import OllamaEmbeddings

from get_config import config

def get_embeddings_func():
    return OllamaEmbeddings(model=config["embedding_model"])
