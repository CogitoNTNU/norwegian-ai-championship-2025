#!/usr/bin/env python3
"""
Script to install faiss-gpu for CUDA acceleration.
This will uninstall faiss-cpu and install faiss-gpu.
"""

import subprocess
import sys
import torch

def check_cuda():
    """Check if CUDA is available."""
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return cuda_available

def install_faiss_gpu():
    """Install faiss-gpu and remove faiss-cpu."""
    print("Installing faiss-gpu...")

    try:
        # Remove faiss-cpu first
        print("Removing faiss-cpu...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "faiss-cpu", "-y"],
                      check=True)

        # Install faiss-gpu
        print("Installing faiss-gpu...")
        subprocess.run([sys.executable, "-m", "pip", "install", "faiss-gpu"],
                      check=True)

        print("✅ faiss-gpu installed successfully!")
        print("You can now use GPU acceleration in your RAG pipeline.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print("Try manually: pip uninstall faiss-cpu && pip install faiss-gpu")

def main():
    print("FAISS GPU Installation Helper")
    print("=" * 40)

    if not check_cuda():
        print("❌ CUDA not available. GPU acceleration will not work.")
        return

    print("\nThis will:")
    print("1. Uninstall faiss-cpu")
    print("2. Install faiss-gpu")
    print("3. Enable GPU acceleration for FAISS operations")

    response = input("\nProceed? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        install_faiss_gpu()
    else:
        print("Installation cancelled.")

if __name__ == "__main__":
    main()
