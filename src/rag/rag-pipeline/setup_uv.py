#!/usr/bin/env python3
"""
UV-optimized setup script for the Emergency Healthcare RAG system.
This version assumes you're using uv as your package manager.
"""

import sys
import subprocess
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def check_uv_installed():
    """Check if uv is installed and working."""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ uv is available: {result.stdout.strip()}")
            return True
        else:
            print("❌ uv is not working properly")
            return False
    except FileNotFoundError:
        print("❌ uv is not installed. Please install it first:")
        print("   pip install uv")
        print("   or follow: https://github.com/astral-sh/uv")
        return False


def setup_uv_environment():
    """Set up the uv environment and install dependencies."""
    print("📦 Setting up uv environment...")
    try:
        # Initialize uv project if needed
        if not (current_dir / "uv.lock").exists():
            print("Initializing uv project...")
            subprocess.run(["uv", "init", "--no-readme"], cwd=current_dir, check=True)

        # Install dependencies
        print("Installing dependencies with uv...")
        subprocess.run(["uv", "sync"], cwd=current_dir, check=True)

        print("✅ Dependencies installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to setup uv environment: {e}")
        return False


def install_ollama():
    """Install Ollama if not already installed."""
    try:
        # Check if ollama is already installed
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
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


def run_with_uv(script_args):
    """Run a Python script using uv."""
    try:
        cmd = ["uv", "run", "python"] + script_args
        subprocess.run(cmd, cwd=current_dir, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run with uv: {e}")
        return False


def setup_llm_model():
    """Download and setup the LLM model using uv run."""
    print("🤖 Setting up LLM model...")
    try:
        # Create a simple script to test LLM setup
        test_script = current_dir / "test_llm.py"
        test_script.write_text("""
from llm_client import LocalLLMClient
        client = LocalLLMClient("qwen3:8b")
client.ensure_model_available()
print("LLM model ready")
""")

        success = run_with_uv(["test_llm.py"])
        test_script.unlink()  # Clean up

        if success:
            print("✅ LLM model ready")
            return True
        else:
            return False

    except Exception as e:
        print(f"❌ Failed to setup LLM model: {e}")
        return False


def build_document_index():
    """Build the document index using uv run."""
    print("📚 Building document index...")
    try:
        # Create index building script
        index_script = current_dir / "build_index.py"
        index_script.write_text("""
from rag_pipeline import RAGPipeline
from pathlib import Path

current_dir = Path(__file__).parent
base_path = current_dir.parent.parent
topics_dir = base_path / "DM-i-AI-2025/emergency-healthcare-rag/data/topics"
topics_json = base_path / "DM-i-AI-2025/emergency-healthcare-rag/data/topics.json"
index_path = current_dir / "medical_index"

if not topics_dir.exists():
    print(f"Topics directory not found: {topics_dir}")
    exit(1)

if not topics_json.exists():
    print(f"Topics JSON not found: {topics_json}")
    exit(1)

pipeline = RAGPipeline()
pipeline.setup(str(topics_dir), str(topics_json), str(index_path))
print("Document index built successfully")
""")

        success = run_with_uv(["build_index.py"])
        index_script.unlink()  # Clean up

        if success:
            print("✅ Document index built successfully")
            return True
        else:
            return False

    except Exception as e:
        print(f"❌ Failed to build document index: {e}")
        return False


def test_classifier():
    """Test the classifier using uv run."""
    print("🧪 Testing classifier...")
    try:
        test_script = current_dir / "test_classifier.py"
        test_script.write_text("""
from classifier import predict

test_statement = "Testicular torsion is a urological emergency requiring immediate surgical intervention."
statement_is_true, statement_topic = predict(test_statement)

print(f"Test successful!")
print(f"Statement: {test_statement}")
print(f"Predicted: is_true={statement_is_true}, topic={statement_topic}")
""")

        success = run_with_uv(["test_classifier.py"])
        test_script.unlink()  # Clean up

        if success:
            print("✅ Classifier test passed")
            return True
        else:
            return False

    except Exception as e:
        print(f"❌ Classifier test failed: {e}")
        return False


def main():
    """Main setup function optimized for uv."""
    print("🚀 Setting up Emergency Healthcare RAG System (UV Edition)")
    print("=" * 60)

    success = True

    # Step 1: Check uv installation
    print("\\n1. Checking uv installation...")
    if not check_uv_installed():
        return False

    # Step 2: Setup uv environment
    print("\\n2. Setting up uv environment...")
    if not setup_uv_environment():
        success = False

    # Step 3: Install Ollama
    print("\\n3. Setting up Ollama...")
    if not install_ollama():
        success = False

    # Step 4: Setup LLM model
    print("\\n4. Setting up LLM model...")
    if not setup_llm_model():
        success = False

    # Step 5: Build document index
    print("\\n5. Building document index...")
    if not build_document_index():
        success = False

    # Step 6: Test classifier
    print("\\n6. Testing classifier...")
    if not test_classifier():
        success = False

    print("\\n" + "=" * 60)
    if success:
        print("🎉 UV Setup completed successfully!")
        print("\\nNext steps:")
        print(
            "1. Test manually: uv run python -c 'from classifier import predict; print(predict(\"test\"))'"
        )
        print("2. Run integration: uv run python integration.py")
        print(
            "3. Test competition: cd ../../DM-i-AI-2025/emergency-healthcare-rag && python example.py"
        )
    else:
        print("❌ Setup failed. Please check the errors above.")

    return success


if __name__ == "__main__":
    main()
