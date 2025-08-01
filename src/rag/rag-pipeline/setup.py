#!/usr/bin/env python3
"""
Setup script for the Emergency Healthcare RAG system.
Run this to initialize everything before using the classifier.
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def install_ollama():
    """Install Ollama if not already installed."""
    import subprocess
    
    try:
        # Check if ollama is already installed
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is already installed")
            return True
    except FileNotFoundError:
        pass
    
    print("📥 Installing Ollama...")
    try:
        # Install ollama
        install_cmd = "curl -fsSL https://ollama.ai/install.sh | sh"
        result = subprocess.run(install_cmd, shell=True, check=True)
        print("✅ Ollama installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Ollama: {e}")
        return False

def setup_llm_model():
    """Download and setup the LLM model."""
    from llm_client import LocalLLMClient
    
    print("🤖 Setting up LLM model...")
    try:
        client = LocalLLMClient("llama3.1:8b")
        client.ensure_model_available()
        print("✅ LLM model ready")
        return True
    except Exception as e:
        print(f"❌ Failed to setup LLM model: {e}")
        return False

def build_document_index():
    """Build the document index from medical articles."""
    from rag_pipeline import RAGPipeline
    
    print("📚 Building document index...")
    try:
        # Initialize pipeline
        pipeline = RAGPipeline()
        
        # Set up paths
        base_path = current_dir.parent.parent
        topics_dir = base_path / "DM-i-AI-2025/emergency-healthcare-rag/data/topics"
        topics_json = base_path / "DM-i-AI-2025/emergency-healthcare-rag/data/topics.json"
        index_path = current_dir / "medical_index"
        
        # Check if paths exist
        if not topics_dir.exists():
            print(f"❌ Topics directory not found: {topics_dir}")
            return False
        
        if not topics_json.exists():
            print(f"❌ Topics JSON not found: {topics_json}")
            return False
        
        # Build index
        pipeline.setup(str(topics_dir), str(topics_json), str(index_path))
        print("✅ Document index built successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to build document index: {e}")
        return False

def test_classifier():
    """Test the classifier with a sample statement."""
    print("🧪 Testing classifier...")
    try:
        from classifier import predict
        
        # Test with a simple medical statement
        test_statement = "Testicular torsion is a urological emergency requiring immediate surgical intervention."
        
        statement_is_true, statement_topic = predict(test_statement)
        
        print(f"✅ Test successful!")
        print(f"   Statement: {test_statement}")
        print(f"   Predicted: is_true={statement_is_true}, topic={statement_topic}")
        return True
        
    except Exception as e:
        print(f"❌ Classifier test failed: {e}")
        return False

def run_training_evaluation():
    """Run evaluation on a few training samples."""
    print("📊 Running training evaluation...")
    try:
        from rag_pipeline import RAGPipeline
        
        pipeline = RAGPipeline()
        
        # Set up paths
        base_path = current_dir.parent.parent
        topics_dir = base_path / "DM-i-AI-2025/emergency-healthcare-rag/data/topics"
        topics_json = base_path / "DM-i-AI-2025/emergency-healthcare-rag/data/topics.json"
        index_path = current_dir / "medical_index"
        
        # Setup pipeline (will load existing index)
        pipeline.setup(str(topics_dir), str(topics_json), str(index_path))
        
        # Evaluate on training data
        train_statements = base_path / "DM-i-AI-2025/emergency-healthcare-rag/data/train/statements"
        train_answers = base_path / "DM-i-AI-2025/emergency-healthcare-rag/data/train/answers"
        
        results = pipeline.evaluate_on_training_data(
            str(train_statements), 
            str(train_answers), 
            max_samples=5
        )
        
        print("✅ Training evaluation completed:")
        print(f"   Binary accuracy: {results['binary_accuracy']:.2f}")
        print(f"   Topic accuracy: {results['topic_accuracy']:.2f}")
        print(f"   Both correct: {results['both_accuracy']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training evaluation failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Emergency Healthcare RAG System")
    print("=" * 50)
    
    success = True
    
    # Step 1: Install dependencies
    print("\n1. Installing Python dependencies...")
    import subprocess
    try:
        # Try uv first, fallback to pip
        result = subprocess.run(['uv', '--version'], capture_output=True)
        if result.returncode == 0:
            print("Using uv package manager...")
            subprocess.run(['uv', 'sync'], check=True)
        else:
            print("uv not found, using pip...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                          check=True)
        print("✅ Python dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        success = False
    
    # Step 2: Install Ollama
    print("\n2. Setting up Ollama...")
    if not install_ollama():
        success = False
    
    # Step 3: Setup LLM model
    print("\n3. Setting up LLM model...")
    if not setup_llm_model():
        success = False
    
    # Step 4: Build document index
    print("\n4. Building document index...")
    if not build_document_index():
        success = False
    
    # Step 5: Test classifier
    print("\n5. Testing classifier...")
    if not test_classifier():
        success = False
    
    # Step 6: Run training evaluation
    print("\n6. Running training evaluation...")
    if not run_training_evaluation():
        print("⚠️  Training evaluation failed, but system may still work")
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Test with: python -c 'from classifier import predict; print(predict(\"test statement\"))'")
        print("2. Replace the model.py in competition directory with classifier.py")
        print("3. Update the competition's model.py to import from src/rag/classifier")
    else:
        print("❌ Setup failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    main()
