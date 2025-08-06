#!/usr/bin/env python3
"""
Final medical embedding training script - simplified for compatibility.
Creates a model that can be loaded and used in notebooks.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def check_cuda():
    """Check if CUDA is available and working."""
    try:
        import torch
        
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Test CUDA tensor operations
            x = torch.rand(100, 100).cuda()
            y = torch.rand(100, 100).cuda()
            z = torch.matmul(x, y)
            logger.info(f"CUDA tensor test successful on {z.device}")
            
            return "cuda"
        else:
            logger.warning("CUDA not available, using CPU")
            return "cpu"
            
    except Exception as e:
        logger.error(f"Error checking CUDA: {e}")
        return "cpu"

def load_and_prepare_dataset(dataset_path: str):
    """Load dataset and prepare it for training."""
    logger.info(f"Loading dataset from {dataset_path}")
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = data['examples']
    logger.info(f"Loaded {len(examples)} examples")
    
    # Show dataset stats
    if 'token_stats' in data:
        stats = data['token_stats']
        logger.info(f"Dataset quality:")
        logger.info(f"  Average similarity: {stats.get('avg_similarity', 'N/A'):.3f}")
        logger.info(f"  Average positive tokens: {stats.get('avg_positive_tokens', 'N/A'):.0f}")
    
    return examples

def create_training_examples(examples):
    """Convert dataset to InputExamples for training."""
    from sentence_transformers import InputExample
    
    train_examples = []
    
    for example in examples:
        anchor = example['anchor']
        positive = example['positive']
        negatives = example.get('negatives', [])
        
        # Create positive pair
        train_examples.append(InputExample(
            texts=[anchor, positive], 
            label=1.0
        ))
        
        # Create negative pairs
        for negative in negatives:
            train_examples.append(InputExample(
                texts=[anchor, negative], 
                label=0.0
            ))
    
    logger.info(f"Created {len(train_examples)} training examples")
    
    # Count positives and negatives
    positives = sum(1 for ex in train_examples if ex.label == 1.0)
    negatives = sum(1 for ex in train_examples if ex.label == 0.0)
    logger.info(f"  Positive pairs: {positives}")
    logger.info(f"  Negative pairs: {negatives}")
    
    return train_examples

def train_model(examples, device="cuda", output_dir=None):
    """Train the medical embedding model using simple approach."""
    from sentence_transformers import SentenceTransformer, InputExample
    from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    from torch.utils.data import DataLoader
    import torch
    
    logger.info("Starting model training...")
    
    # Create timestamped output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        output_dir = str(models_dir / f"pubmedbert-medical-final_{timestamp}")
    
    # Load base model - PubMedBERT trained from scratch on medical literature
    model_id = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    logger.info(f"Loading base model: {model_id}")
    
    model = SentenceTransformer(model_id, device=device)
    logger.info(f"Model loaded successfully on {device}")
    
    # Convert dataset to training examples
    train_examples = create_training_examples(examples)
    
    # Split into train/eval (90/10)
    split_idx = int(len(train_examples) * 0.9)
    train_data = train_examples[:split_idx]
    eval_data = train_examples[split_idx:]
    
    logger.info(f"Train examples: {len(train_data)}")
    logger.info(f"Eval examples: {len(eval_data)}")
    
    # Create loss function with Matryoshka representation learning
    matryoshka_dims = [768, 512, 256, 128, 64]
    base_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(model, base_loss, matryoshka_dims)
    
    logger.info(f"Using Matryoshka dimensions: {matryoshka_dims}")
    
    # Create evaluator
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        eval_data, name='medical-eval'
    )
    
    # Training parameters optimized for medical domain - 100 EPOCHS
    if device == "cuda":
        batch_size = 16  # Conservative for 8GB VRAM
        epochs = 100  # Maximum training for superior medical performance
        warmup_steps = 400  # Increased warmup for longer training
        use_amp = True
    else:
        batch_size = 8
        epochs = 50  # Half for CPU training
        warmup_steps = 200
        use_amp = False
    
    # Create DataLoader
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    
    # Monitor GPU memory
    if device == "cuda":
        memory_before = torch.cuda.memory_allocated() / 1e9
        logger.info(f"GPU memory before training: {memory_before:.2f} GB")
    
    # Start training using the fit method
    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Batch size: {batch_size}, Warmup steps: {warmup_steps}")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=len(train_dataloader),  # Evaluate each epoch
        warmup_steps=warmup_steps,
        output_path=output_dir,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={'lr': 2e-5},
        weight_decay=0.01,
        scheduler='WarmupLinear',
        save_best_model=True,
        show_progress_bar=True,
        use_amp=use_amp
    )
    
    # Monitor GPU memory after training
    if device == "cuda":
        memory_after = torch.cuda.memory_allocated() / 1e9
        logger.info(f"GPU memory after training: {memory_after:.2f} GB")
        torch.cuda.empty_cache()
    
    logger.info(f"Model training completed! Saved to: {output_dir}")
    
    # Save training info
    training_info = {
        'base_model': model_id,
        'dataset_path': 'improved_medical_embedding_dataset.json',
        'training_timestamp': datetime.now().isoformat(),
        'device': device,
        'matryoshka_dimensions': matryoshka_dims,
        'training_examples': len(train_data),
        'eval_examples': len(eval_data),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': 2e-5,
        'training_method': 'sentence_transformers_fit_api',
        'output_dir': output_dir,
        'model_loadable': True,
        'usage_example': f"model = SentenceTransformer('{output_dir}')"
    }
    
    with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=2)
    
    return output_dir

def test_model(output_dir, device="cuda"):
    """Test the trained model to ensure it can be loaded."""
    logger.info("Testing trained model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load the trained model
        model = SentenceTransformer(output_dir, device=device)
        
        # Test medical queries
        test_texts = [
            "Patient presents with chest pain and shortness of breath",
            "Acute myocardial infarction with ST elevation",
            "The patient has a history of hypertension and diabetes"
        ]
        
        # Generate embeddings
        embeddings = model.encode(test_texts)
        logger.info(f"Successfully generated embeddings with shape: {embeddings.shape}")
        
        # Test different dimensions
        for dim in [768, 512, 256, 128, 64]:
            model_dim = SentenceTransformer(output_dir, truncate_dim=dim)
            embeddings_dim = model_dim.encode(test_texts)
            logger.info(f"Dimension {dim}: embeddings shape {embeddings_dim.shape}")
        
        logger.info("✅ Model testing successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model testing failed: {e}")
        return False

def main():
    """Main training function."""
    logger.info("Starting Final Medical Embedding Training")
    logger.info("=" * 80)
    
    # Check CUDA
    device = check_cuda()
    
    # Load dataset
    dataset_path = "improved_medical_embedding_dataset.json"
    
    try:
        # Load dataset
        examples = load_and_prepare_dataset(dataset_path)
        
        # Train model
        output_dir = train_model(examples, device=device)
        
        # Test the trained model
        model_works = test_model(output_dir, device=device)
        
        if model_works:
            logger.info("=" * 80)
            logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Model saved to: {output_dir}")
            logger.info(f"Device used: {device.upper()}")
            logger.info(f"Training examples: {len(examples)}")
            
            print(f"\nUsage Instructions:")
            print(f"```python")
            print(f"from sentence_transformers import SentenceTransformer")
            print(f"")
            print(f"# Load the trained model")
            print(f"model = SentenceTransformer('{output_dir}')")
            print(f"")
            print(f"# Generate embeddings")
            print(f"embeddings = model.encode(['your medical text here'])")
            print(f"")
            print(f"# Use different Matryoshka dimensions")
            print(f"model_256 = SentenceTransformer('{output_dir}', truncate_dim=256)")
            print(f"model_128 = SentenceTransformer('{output_dir}', truncate_dim=128)")
            print(f"```")
            
            print(f"\nModel Details:")
            print(f"  - Base model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
            print(f"  - Training: 100 epochs for maximum medical performance")
            print(f"  - Matryoshka dimensions: [768, 512, 256, 128, 64]")
            print(f"  - Training method: MultipleNegativesRankingLoss + MatryoshkaLoss")
            print(f"  - Dataset: Improved medical embeddings (semantic similarity based)")
            print(f"  - Compatible with: Jupyter notebooks, sentence-transformers")
            
            return True
        else:
            logger.error("❌ Model training completed but testing failed")
            return False
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)