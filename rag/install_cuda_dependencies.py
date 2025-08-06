#!/usr/bin/env python3
"""
Install CUDA-enabled PyTorch and dependencies for medical embedding training.
This script ensures proper CUDA setup for the training pipeline.
"""

import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run a command and log the result."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"Success: {description}")
        if result.stdout:
            logger.info(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Error: {e.stderr}")
        return False

def install_cuda_pytorch():
    """Install CUDA-enabled PyTorch."""
    logger.info("Installing CUDA-enabled PyTorch...")
    
    # First, uninstall any existing torch packages
    run_command(
        "uv pip uninstall torch torchvision torchaudio -y",
        "Removing existing PyTorch packages"
    )
    
    # Install CUDA-enabled PyTorch
    success = run_command(
        "uv pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall",
        "Installing CUDA PyTorch"
    )
    
    if not success:
        logger.error("Failed to install CUDA PyTorch")
        return False
    
    return True

def install_faiss_gpu():
    """Install FAISS GPU version."""
    logger.info("Installing FAISS GPU...")
    
    # Remove CPU version
    run_command(
        "uv pip uninstall faiss-cpu -y",
        "Removing FAISS CPU"
    )
    
    # Install GPU version with specific version
    success = run_command(
        "uv pip install faiss-gpu==1.7.2",
        "Installing FAISS GPU"
    )
    
    return success

def test_cuda_installation():
    """Test if CUDA installation is working."""
    logger.info("Testing CUDA installation...")
    
    test_script = '''
import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Test tensor operations
    try:
        x = torch.rand(5, 3).cuda()
        y = torch.rand(5, 3).cuda()
        z = x + y
        print("✅ CUDA tensor operations working!")
        print(f"Test tensor device: {z.device}")
        
        # Test memory allocation
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
    except Exception as e:
        print(f"❌ CUDA tensor operations failed: {e}")
        sys.exit(1)
else:
    print("❌ CUDA not available!")
    sys.exit(1)

# Test FAISS GPU
try:
    import faiss
    print(f"FAISS version: {faiss.__version__}")
    
    # Check if GPU resources are available
    if faiss.get_num_gpus() > 0:
        print(f"✅ FAISS GPU available: {faiss.get_num_gpus()} GPUs")
    else:
        print("⚠️  FAISS GPU not available, falling back to CPU")
        
except ImportError as e:
    print(f"❌ FAISS import failed: {e}")
    
print("\\n🎉 CUDA setup verification complete!")
'''
    
    success = run_command(
        f'uv run python -c "{test_script}"',
        "Testing CUDA setup"
    )
    
    return success

def main():
    """Main installation function."""
    logger.info("🚀 Starting CUDA dependencies installation...")
    
    # Check if NVIDIA GPU is available
    try:
        result = subprocess.run("nvidia-smi", capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("❌ NVIDIA GPU not detected. Please ensure NVIDIA drivers are installed.")
            return False
    except FileNotFoundError:
        logger.error("❌ nvidia-smi not found. Please install NVIDIA drivers.")
        return False
    
    # Install CUDA PyTorch
    if not install_cuda_pytorch():
        return False
    
    # Install FAISS GPU
    if not install_faiss_gpu():
        logger.warning("⚠️  FAISS GPU installation failed, will use CPU version")
    
    # Test installation
    if not test_cuda_installation():
        logger.error("❌ CUDA setup test failed")
        return False
    
    logger.info("✅ CUDA dependencies installed successfully!")
    logger.info("You can now run the medical embedding training with CUDA support.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)