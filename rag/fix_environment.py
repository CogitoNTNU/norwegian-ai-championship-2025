#!/usr/bin/env python3
"""
Fix environment dependency conflicts for the RAG pipeline.
This script resolves numpy/sklearn/transformers/torchvision version issues.
"""

import subprocess
import sys
import os

def run_command(cmd, check=True):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        return False

def fix_environment():
    """Fix the environment dependencies."""
    print("🔧 Fixing Environment Dependencies")
    print("=" * 50)

    # Check if we're in UV environment
    if "VIRTUAL_ENV" in os.environ:
        print(f"✅ Using virtual environment: {os.environ['VIRTUAL_ENV']}")
    else:
        print("⚠️  Not in virtual environment, proceeding anyway...")

    # Fix numpy/sklearn compatibility
    print("\n1. Fixing numpy/sklearn compatibility...")
    commands = [
        "pip uninstall numpy scikit-learn -y",
        "pip install numpy==1.24.4",  # Compatible version
        "pip install scikit-learn==1.3.2",  # Compatible version
        "pip install --upgrade --force-reinstall sentence-transformers",
        "pip install --upgrade --force-reinstall transformers",
    ]

    for cmd in commands:
        if not run_command(cmd, check=False):
            print(f"⚠️  Command failed, continuing: {cmd}")

    # Fix torchvision compatibility
    print("\n2. Fixing PyTorch/torchvision compatibility...")
    torch_commands = [
        "pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121",
    ]

    for cmd in torch_commands:
        if not run_command(cmd, check=False):
            print(f"⚠️  PyTorch install failed, trying alternative: {cmd}")
            # Fallback to CPU version
            run_command("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu", check=False)

    print("\n3. Reinstalling core dependencies...")
    core_deps = [
        "pip install --upgrade faiss-cpu",  # Keep CPU version for Python 3.11
        "pip install --upgrade tqdm",
        "pip install --upgrade pathlib",
    ]

    for cmd in core_deps:
        run_command(cmd, check=False)

    print("\n✅ Environment fix completed!")
    print("\nTo test the fix, run:")
    print("  python -c \"import torch; import sentence_transformers; import faiss; print('✅ All imports successful')\"")

def test_imports():
    """Test if critical imports work."""
    print("\n🧪 Testing imports...")

    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except Exception as e:
        print(f"❌ PyTorch: {e}")

    try:
        import sentence_transformers
        print(f"✅ sentence-transformers {sentence_transformers.__version__}")
    except Exception as e:
        print(f"❌ sentence-transformers: {e}")

    try:
        import faiss
        print(f"✅ FAISS")
    except Exception as e:
        print(f"❌ FAISS: {e}")

    try:
        import numpy as np
        import sklearn
        print(f"✅ NumPy {np.__version__}, scikit-learn {sklearn.__version__}")
    except Exception as e:
        print(f"❌ NumPy/sklearn: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true", help="Only test imports, don't fix")
    args = parser.parse_args()

    if args.test_only:
        test_imports()
    else:
        print("This will reinstall some packages to fix compatibility issues.")
        response = input("Continue? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            fix_environment()
            test_imports()
        else:
            print("Cancelled.")
