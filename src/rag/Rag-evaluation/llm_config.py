#!/usr/bin/env python3
"""
Configuration for using local LLM models (via Ollama) instead of Google API.
This replaces the need for Google API keys.
"""

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import os

class LocalLLMConfig:
    """Configuration class for local LLM models via Ollama."""
    
    def __init__(self, model_name="qwen3:8b", embedding_model="nomic-embed-text"):
        self.model_name = model_name
        self.embedding_model = embedding_model
        
    def get_llm(self):
        """Get the LLM instance."""
        return Ollama(model=self.model_name)
    
    def get_embeddings(self):
        """Get the embeddings instance."""
        return OllamaEmbeddings(model=self.embedding_model)
    
    def ensure_models_available(self):
        """Ensure both models are pulled and available."""
        import subprocess
        
        models_to_pull = [self.model_name, self.embedding_model]
        
        for model in models_to_pull:
            try:
                print(f"Checking if {model} is available...")
                result = subprocess.run(['ollama', 'list'], 
                                      capture_output=True, text=True)
                
                if model not in result.stdout:
                    print(f"Pulling {model}...")
                    subprocess.run(['ollama', 'pull', model], check=True)
                    print(f"✅ {model} pulled successfully")
                else:
                    print(f"✅ {model} is already available")
                    
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to pull {model}: {e}")
                return False
        
        return True

# Default configuration
DEFAULT_CONFIG = LocalLLMConfig()

def setup_local_llm_environment():
    """Set up environment to use local LLM models without Google API."""
    # Remove any requirement for Google API key
    if 'GOOGLE_API_KEY' in os.environ:
        del os.environ['GOOGLE_API_KEY']
    
    # Set up Ollama environment
    os.environ['USER_AGENT'] = 'rag-evaluator-local-llm/1.0'
    
    # Disable LangSmith tracing (optional)
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'
    
    print("🤖 Environment configured for local LLM models")
    return DEFAULT_CONFIG

if __name__ == "__main__":
    config = setup_local_llm_environment()
    success = config.ensure_models_available()
    
    if success:
        print("🎉 Local LLM configuration ready!")
        print(f"LLM Model: {config.model_name}")
        print(f"Embedding Model: {config.embedding_model}")
    else:
        print("❌ Failed to set up local LLM models")
