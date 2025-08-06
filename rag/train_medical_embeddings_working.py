#!/usr/bin/env python3
"""
Working medical embedding training script compatible with current environment.
Uses improved dataset and works with sentence-transformers 2.7.0
"""

import os
import sys
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

def load_dataset(dataset_path: str):
    """Load the improved medical dataset."""
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

def create_sentence_pairs(examples):
    """Convert triplet format to sentence pair format for training."""
    from datasets import Dataset
    
    pairs = []
    
    for example in examples:
        anchor = example['anchor']
        positive = example['positive']
        negatives = example.get('negatives', [])
        
        # Add positive pair
        pairs.append({
            'sentence1': anchor,
            'sentence2': positive,
            'label': 1.0
        })
        
        # Add negative pairs
        for negative in negatives:
            pairs.append({
                'sentence1': anchor,
                'sentence2': negative,
                'label': 0.0
            })
    
    logger.info(f"Created {len(pairs)} training pairs")
    
    # Count positives and negatives
    positives = sum(1 for p in pairs if p['label'] == 1.0)
    negatives = sum(1 for p in pairs if p['label'] == 0.0)
    logger.info(f"  Positive pairs: {positives}")
    logger.info(f"  Negative pairs: {negatives}")
    
    return Dataset.from_list(pairs)

def train_model(dataset, device="cuda", output_dir=None):
    """Train the medical embedding model using the older API."""
    from sentence_transformers import SentenceTransformer, InputExample
    from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    import torch
    
    logger.info("Initializing model training...")
    
    # Create timestamped output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        output_dir = str(models_dir / f"biobert-medical-embeddings_{timestamp}")
    
    # Load base model
    model_id = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    logger.info(f"Loading base model: {model_id}")
    
    try:
        model = SentenceTransformer(model_id, device=device)
        logger.info(f"Model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Trying with CPU...")
        model = SentenceTransformer(model_id, device="cpu")
        device = "cpu"
    
    # Convert dataset to list and split
    pairs = list(dataset)
    split_idx = int(len(pairs) * 0.9)  # 90% train, 10% eval
    train_data = pairs[:split_idx]
    eval_data = pairs[split_idx:]
    
    # Convert to InputExamples
    train_examples = []
    eval_examples = []
    
    for pair in train_data:
        train_examples.append(InputExample(
            texts=[pair['sentence1'], pair['sentence2']], 
            label=float(pair['label'])
        ))
    
    for pair in eval_data:
        eval_examples.append(InputExample(
            texts=[pair['sentence1'], pair['sentence2']], 
            label=float(pair['label'])
        ))
    
    logger.info(f"Train examples: {len(train_examples)}")
    logger.info(f"Eval examples: {len(eval_examples)}")
    
    # Create loss function with Matryoshka representation learning
    matryoshka_dims = [768, 512, 256, 128, 64]
    base_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(model, base_loss, matryoshka_dims)
    
    logger.info(f"Using Matryoshka dimensions: {matryoshka_dims}")
    
    # Create evaluator
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        eval_examples, name='medical-eval'
    )
    
    # Training parameters optimized for medical domain
    if device == "cuda":
        train_batch_size = 8  # Conservative for 8GB VRAM
        warmup_steps = int(len(train_examples) * 0.1)
    else:
        train_batch_size = 16
        warmup_steps = int(len(train_examples) * 0.1)
    
    # Start training
    logger.info("Starting training...")
    if device == "cuda":
        memory_before = torch.cuda.memory_allocated() / 1e9
        logger.info(f"GPU memory before training: {memory_before:.2f} GB")
    
    # Create DataLoader for training
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
    
    # Use the old fit() method
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=4,
        evaluation_steps=500,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        save_best_model=True,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={'lr': 2e-5},
        show_progress_bar=True,
    )
    
    if device == "cuda":
        memory_after = torch.cuda.memory_allocated() / 1e9
        logger.info(f"GPU memory after training: {memory_after:.2f} GB")
        torch.cuda.empty_cache()
    
    logger.info(f"Model saved to: {output_dir}")
    
    # Save training info
    training_info = {
        'base_model': model_id,
        'dataset_path': 'improved_medical_embedding_dataset.json',
        'training_timestamp': datetime.now().isoformat(),
        'device': device,
        'matryoshka_dimensions': matryoshka_dims,
        'training_examples': len(train_examples),
        'eval_examples': len(eval_examples),
        'epochs': 4,
        'batch_size': train_batch_size,
        'learning_rate': 2e-5
    }
    
    with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=2)
    
    return output_dir

def main():
    """Main training function."""
    logger.info("Starting Medical Embedding Training")
    logger.info("=" * 60)
    
    # Check CUDA
    device = check_cuda()
    
    # Load dataset
    dataset_path = "improved_medical_embedding_dataset.json"
    
    try:
        examples = load_dataset(dataset_path)
        dataset = create_sentence_pairs(examples)
        
        # Train model
        output_dir = train_model(dataset, device=device)
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"Device used: {device.upper()}")
        logger.info(f"Dataset: Improved medical embeddings ({len(examples)} examples)")
        
        print(f"\nTo use your trained model:")
        print(f"```python")
        print(f"from sentence_transformers import SentenceTransformer")
        print(f"model = SentenceTransformer('{output_dir}')")
        print(f"embeddings = model.encode(['your medical text here'])")
        print(f"```")
        
        print(f"\nFor different Matryoshka dimensions:")
        print(f"model = SentenceTransformer('{output_dir}', truncate_dim=256)")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)