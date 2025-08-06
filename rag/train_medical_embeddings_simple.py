#!/usr/bin/env python3
"""
Simple medical embedding model fine-tuning script
Uses your improved_medical_embedding_dataset.json for training
"""

import json
import os
import sys
import logging
import torch
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

def install_requirements():
    """Install required packages if not available."""
    try:
        import sentence_transformers
        logger.info("sentence-transformers already installed")
    except ImportError:
        logger.info("Installing sentence-transformers...")
        os.system("pip install sentence-transformers")

def load_medical_dataset(dataset_path: str) -> List[Dict]:
    """Load your medical embedding dataset."""
    logger.info(f"Loading dataset from {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = data.get('examples', [])
    logger.info(f"Loaded {len(examples)} training examples")
    
    return examples

def prepare_training_data(examples: List[Dict]) -> List[Tuple[str, str]]:
    """Convert examples to anchor-positive pairs for training."""
    training_pairs = []
    
    for example in examples:
        anchor = example.get('anchor', '')
        positive = example.get('positive', '')
        
        if anchor and positive:
            training_pairs.append((anchor, positive))
    
    logger.info(f"Prepared {len(training_pairs)} training pairs")
    return training_pairs

def create_sentence_transformers_dataset(training_pairs: List[Tuple[str, str]]):
    """Create dataset compatible with sentence-transformers."""
    from sentence_transformers import InputExample
    
    train_examples = []
    for anchor, positive in training_pairs:
        train_examples.append(InputExample(texts=[anchor, positive]))
    
    return train_examples

def train_medical_embeddings(
    dataset_path: str = "improved_medical_embedding_dataset.json",
    base_model: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    output_dir: str = None,
    epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    """Train medical embeddings model using your dataset."""
    
    # Install requirements
    install_requirements()
    
    # Import after installation
    from sentence_transformers import SentenceTransformer, losses, evaluation
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    from torch.utils.data import DataLoader
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"biobert-medical-embeddings_{timestamp}"
    
    logger.info(f"Output directory: {output_dir}")
    
    # Load dataset
    examples = load_medical_dataset(dataset_path)
    if not examples:
        logger.error("No examples found in dataset!")
        return
    
    # Prepare training data
    training_pairs = prepare_training_data(examples)
    if not training_pairs:
        logger.error("No training pairs created!")
        return
    
    # Create train/validation split
    split_idx = int(0.9 * len(training_pairs))
    train_pairs = training_pairs[:split_idx]
    val_pairs = training_pairs[split_idx:]
    
    logger.info(f"Training pairs: {len(train_pairs)}")
    logger.info(f"Validation pairs: {len(val_pairs)}")
    
    # Load base model
    logger.info(f"Loading base model: {base_model}")
    model = SentenceTransformer(base_model, device=device)
    
    # Create training examples
    train_examples = create_sentence_transformers_dataset(train_pairs)
    
    # Create data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Define loss function - Multiple Negatives Ranking Loss is perfect for your data
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Create evaluator for validation
    if val_pairs:
        val_examples = create_sentence_transformers_dataset(val_pairs)
        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
            val_examples, name='medical-eval'
        )
    else:
        evaluator = None
    
    # Calculate warmup steps
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)
    
    logger.info("Starting training...")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Warmup steps: {warmup_steps}")
    
    # Training using the fit() method (older sentence-transformers API)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=500,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        save_best_model=True,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={'lr': learning_rate},
        show_progress_bar=True
    )
    
    logger.info(f"Model saved to: {output_dir}")
    
    # Save training info
    training_info = {
        'base_model': base_model,
        'dataset_path': dataset_path,
        'training_timestamp': datetime.now().isoformat(),
        'device': device,
        'training_examples': len(train_examples),
        'validation_examples': len(val_pairs) if val_pairs else 0,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'warmup_steps': warmup_steps,
        'output_directory': output_dir
    }
    
    info_path = os.path.join(output_dir, 'training_info.json')
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    logger.info(f"Training info saved to: {info_path}")
    
    return output_dir

def test_trained_model(model_path: str):
    """Test the trained model with some examples."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    
    logger.info(f"Loading trained model from: {model_path}")
    model = SentenceTransformer(model_path)
    
    # Test sentences
    test_sentences = [
        "Patient presents with acute chest pain and shortness of breath",
        "Myocardial infarction with ST elevation on ECG", 
        "Normal cardiac function on echocardiogram",
        "Sepsis with elevated lactate levels"
    ]
    
    logger.info("Testing trained model:")
    embeddings = model.encode(test_sentences)
    
    # Calculate similarities
    similarities = cos_sim(embeddings, embeddings)
    
    print("\nSimilarity matrix:")
    for i, sent1 in enumerate(test_sentences):
        print(f"\n{i+1}. {sent1[:50]}...")
        for j, sent2 in enumerate(test_sentences):
            if i != j:
                sim_score = similarities[i][j].item()
                print(f"   vs {j+1}: {sim_score:.4f}")

def save_model_for_reuse(model_path: str, final_output_path: str = None):
    """Save model in a format that can be easily reused."""
    from sentence_transformers import SentenceTransformer
    
    if final_output_path is None:
        final_output_path = f"{model_path}_final"
    
    # Load and re-save the model
    model = SentenceTransformer(model_path)
    model.save(final_output_path)
    
    logger.info(f"Model saved for reuse at: {final_output_path}")
    
    # Create a simple loading script
    loader_script = f"""
# How to load and use your fine-tuned medical embeddings model

from sentence_transformers import SentenceTransformer

# Load your fine-tuned model
model = SentenceTransformer("{final_output_path}")

# Example usage
medical_texts = [
    "Patient presents with chest pain",
    "Acute myocardial infarction diagnosed",
    "Normal cardiac examination findings"
]

# Get embeddings
embeddings = model.encode(medical_texts)
print(f"Generated embeddings shape: {{embeddings.shape}}")

# Calculate similarities
from sentence_transformers.util import cos_sim
similarities = cos_sim(embeddings, embeddings)
print("Similarity matrix:")
print(similarities)
"""
    
    with open(f"{final_output_path}_usage_example.py", 'w') as f:
        f.write(loader_script)
    
    return final_output_path

def main():
    """Main training pipeline."""
    logger.info("Starting medical embeddings training pipeline")
    
    # Check if dataset exists
    dataset_path = "improved_medical_embedding_dataset.json"
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        logger.error("Please make sure you have created the medical embedding dataset first")
        return
    
    # Train the model
    try:
        model_path = train_medical_embeddings(
            dataset_path=dataset_path,
            base_model="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
            epochs=100,
            batch_size=32,
            learning_rate=2e-5
        )
        
        # Test the trained model
        test_trained_model(model_path)
        
        # Save for easy reuse
        final_path = save_model_for_reuse(model_path)
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Final model saved at: {final_path}")
        logger.info(f"Usage example saved at: {final_path}_usage_example.py")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
