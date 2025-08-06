#!/usr/bin/env python3
"""
Medical embedding training script following the notebook format.
Compatible with sentence-transformers 2.7.0 and uses the fit() API.
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
    """Load dataset and prepare it in notebook format."""
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
    
    # Convert to notebook format: anchor, positive, id
    formatted_examples = []
    for i, example in enumerate(examples):
        formatted_examples.append({
            'anchor': example['anchor'],
            'positive': example['positive'],
            'id': i
        })
    
    logger.info(f"Formatted {len(formatted_examples)} examples for training")
    return formatted_examples

def prepare_dataset_objects(examples):
    """Prepare dataset objects for training and evaluation."""
    from datasets import Dataset
    
    # Create datasets
    dataset = Dataset.from_list(examples)
    
    # Shuffle and split (90/10)
    dataset = dataset.shuffle()
    dataset = dataset.train_test_split(test_size=0.1)
    
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    logger.info(f"Train dataset: {len(train_dataset)} examples")
    logger.info(f"Test dataset: {len(test_dataset)} examples")
    
    return train_dataset, test_dataset

def prepare_evaluation_data(train_dataset, test_dataset):
    """Prepare evaluation data structures following the notebook format."""
    from datasets import concatenate_datasets
    
    # Combine datasets for corpus
    corpus_dataset = concatenate_datasets([train_dataset, test_dataset])
    
    # Create corpus: maps corpus IDs to their text chunks
    corpus = dict(zip(corpus_dataset["id"], corpus_dataset["positive"]))
    
    # Create queries: maps query IDs to their questions (only test set)
    queries = dict(zip(test_dataset["id"], test_dataset["anchor"]))
    
    # Create relevance mapping - each query maps to its positive document
    relevant_docs = {}
    for q_id, example_id in zip(test_dataset["id"], test_dataset["id"]):
        # Find matching corpus document
        relevant_docs[q_id] = [example_id]  # Each query maps to its own positive
    
    logger.info(f"Prepared evaluation data:")
    logger.info(f"  Corpus: {len(corpus)} documents")
    logger.info(f"  Queries: {len(queries)} queries")
    logger.info(f"  Average relevant docs per query: 1.0")
    
    return corpus, queries, relevant_docs

def create_evaluator(corpus, queries, relevant_docs):
    """Create Matryoshka evaluators following the notebook format."""
    from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
    from sentence_transformers.util import cos_sim
    
    # Matryoshka dimensions (large to small)
    matryoshka_dimensions = [768, 512, 256, 128, 64]
    
    # Create evaluators for each dimension
    matryoshka_evaluators = []
    for dim in matryoshka_dimensions:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,
            score_functions={"cosine": cos_sim},
        )
        matryoshka_evaluators.append(ir_evaluator)
    
    # Create sequential evaluator
    evaluator = SequentialEvaluator(matryoshka_evaluators)
    
    logger.info(f"Created evaluator with dimensions: {matryoshka_dimensions}")
    return evaluator

def evaluate_base_model(model, evaluator):
    """Evaluate base model performance."""
    logger.info("Evaluating base model performance...")
    
    base_results = evaluator(model)
    
    # Debug: Print what base_results looks like
    logger.info(f"Base results type: {type(base_results)}")
    if hasattr(base_results, 'keys'):
        logger.info(f"Base results keys: {list(base_results.keys())[:10]}")  # Show first 10 keys
    
    # Print results in notebook format
    print("\nBase Model Evaluation Results")
    print("-" * 85)
    print(f"{'Metric':15} {'768d':>12} {'512d':>12} {'256d':>12} {'128d':>12} {'64d':>12}")
    print("-" * 85)
    
    # Matryoshka dimensions
    matryoshka_dimensions = [768, 512, 256, 128, 64]
    
    # Metrics to display
    metrics = [
        'ndcg@10',
        'mrr@10', 
        'map@100',
        'accuracy@1',
        'accuracy@3',
        'accuracy@5',
        'accuracy@10',
        'precision@1',
        'precision@3',
        'precision@5',
        'precision@10',
        'recall@1',
        'recall@3',
        'recall@5',
        'recall@10'
    ]
    
    # Print each metric
    for metric in metrics:
        values = []
        for dim in matryoshka_dimensions:
            key = f"dim_{dim}_cosine_{metric}"
            if isinstance(base_results, dict):
                values.append(base_results.get(key, 0.0))
            else:
                values.append(0.0)  # Fallback
        
        # Highlight NDCG@10
        metric_name = f"=={metric}==" if metric == "ndcg@10" else metric
        print(f"{metric_name:15}", end="  ")
        for val in values:
            print(f"{val:12.4f}", end=" ")
        print()
    
    # Print sequential score
    print("-" * 85)
    if isinstance(base_results, dict):
        print(f"{'seq_score:'} {base_results.get('sequential_score', 0.0):.4f}")
    else:
        print(f"{'seq_score:'} 0.0000")
    
    return base_results

def train_model(train_dataset, evaluator, device="cuda", output_dir=None):
    """Train the medical embedding model using the notebook approach."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
    import torch
    
    logger.info("Starting model training (notebook style)...")
    
    # Create timestamped output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        output_dir = str(models_dir / f"biobert-medical-embeddings-nb_{timestamp}")
    
    # Load base model following notebook format
    model_id = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    logger.info(f"Loading base model: {model_id}")
    
    # Load model (simpler initialization for older version)
    model = SentenceTransformer(model_id, device=device)
    
    logger.info(f"Model loaded successfully on {device}")
    
    # Create loss functions following notebook format
    matryoshka_dimensions = [768, 512, 256, 128, 64]
    
    # Base loss: MultipleNegativesRankingLoss
    base_loss = MultipleNegativesRankingLoss(model)
    
    # Matryoshka Loss Wrapper
    train_loss = MatryoshkaLoss(
        model, base_loss, matryoshka_dims=matryoshka_dimensions
    )
    
    logger.info(f"Using Matryoshka dimensions: {matryoshka_dimensions}")
    
    # Training parameters adapted from notebook but for older API
    if device == "cuda":
        batch_size = 16  # Conservative for 8GB VRAM
        epochs = 4
        warmup_steps = int(len(train_dataset) * 0.1)
        use_amp = True
    else:
        batch_size = 8
        epochs = 2
        warmup_steps = int(len(train_dataset) * 0.1)
        use_amp = False
    
    # Monitor GPU memory
    if device == "cuda":
        memory_before = torch.cuda.memory_allocated() / 1e9
        logger.info(f"GPU memory before training: {memory_before:.2f} GB")
    
    # Start training using the fit method (compatible with v2.7.0)
    logger.info(f"Starting training for {epochs} epochs...")
    logger.info(f"Batch size: {batch_size}, Warmup steps: {warmup_steps}")
    
    model.fit(
        train_objectives=[(train_dataset, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=max(1, len(train_dataset) // batch_size),  # Evaluate each epoch
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
        'matryoshka_dimensions': matryoshka_dimensions,
        'training_examples': len(train_dataset),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': 2e-5,
        'training_method': 'notebook_style_fit_api',
        'output_dir': output_dir
    }
    
    with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=2)
    
    return output_dir

def evaluate_fine_tuned_model(output_dir, evaluator, device="cuda"):
    """Evaluate the fine-tuned model."""
    from sentence_transformers import SentenceTransformer
    
    logger.info("Evaluating fine-tuned model...")
    
    # Load the fine-tuned model
    fine_tuned_model = SentenceTransformer(
        output_dir, device=device
    )
    
    # Evaluate
    ft_results = evaluator(fine_tuned_model)
    
    # Print results
    print("\nFine-Tuned Model Evaluation Results")
    print("-" * 85)
    print(f"{'Metric':15} {'768d':>12} {'512d':>12} {'256d':>12} {'128d':>12} {'64d':>12}")
    print("-" * 85)
    
    # Matryoshka dimensions
    matryoshka_dimensions = [768, 512, 256, 128, 64]
    
    # Metrics to display
    metrics = [
        'ndcg@10',
        'mrr@10',
        'map@100',
        'accuracy@1',
        'accuracy@3',
        'accuracy@5',
        'accuracy@10',
        'precision@1',
        'precision@3',
        'precision@5',
        'precision@10',
        'recall@1',
        'recall@3',
        'recall@5',
        'recall@10'
    ]
    
    # Print each metric
    for metric in metrics:
        values = []
        for dim in matryoshka_dimensions:
            key = f"dim_{dim}_cosine_{metric}"
            if isinstance(ft_results, dict):
                values.append(ft_results.get(key, 0.0))
            else:
                values.append(0.0)  # Fallback
        
        # Highlight NDCG@10
        metric_name = f"=={metric}==" if metric == "ndcg@10" else metric
        print(f"{metric_name:15}", end="  ")
        for val in values:
            print(f"{val:12.4f}", end=" ")
        print()
    
    # Print sequential score
    print("-" * 85)
    if isinstance(ft_results, dict):
        print(f"{'seq_score:'} {ft_results.get('sequential_score', 0.0):.4f}")
    else:
        print(f"{'seq_score:'} 0.0000")
    
    return ft_results

def main():
    """Main training function following the notebook workflow."""
    logger.info("Starting Medical Embedding Training (Notebook Style)")
    logger.info("=" * 80)
    
    # Check CUDA
    device = check_cuda()
    
    # Load dataset
    dataset_path = "improved_medical_embedding_dataset.json"
    
    try:
        # Step 1: Load and prepare dataset
        examples = load_and_prepare_dataset(dataset_path)
        
        # Step 2: Create dataset objects
        train_dataset, test_dataset = prepare_dataset_objects(examples)
        
        # Step 3: Prepare evaluation data structures
        corpus, queries, relevant_docs = prepare_evaluation_data(train_dataset, test_dataset)
        
        # Step 4: Create evaluator
        evaluator = create_evaluator(corpus, queries, relevant_docs)
        
        # Step 5: Load base model and evaluate
        logger.info("Loading base model for evaluation...")
        from sentence_transformers import SentenceTransformer
        model_id = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
        base_model = SentenceTransformer(model_id, device=device)
        
        base_results = evaluate_base_model(base_model, evaluator)
        
        # Step 6: Train model
        output_dir = train_model(train_dataset, evaluator, device=device)
        
        # Step 7: Evaluate fine-tuned model
        ft_results = evaluate_fine_tuned_model(output_dir, evaluator, device=device)
        
        # Step 8: Success message
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"Device used: {device.upper()}")
        logger.info(f"Training examples: {len(train_dataset)}")
        logger.info(f"Test examples: {len(test_dataset)}")
        
        print(f"\nTo use your trained model:")
        print(f"```python")
        print(f"from sentence_transformers import SentenceTransformer")
        print(f"model = SentenceTransformer('{output_dir}')")
        print(f"embeddings = model.encode(['your medical text here'])")
        print(f"```")
        
        print(f"\nFor different Matryoshka dimensions:")
        print(f"model = SentenceTransformer('{output_dir}', truncate_dim=256)")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)