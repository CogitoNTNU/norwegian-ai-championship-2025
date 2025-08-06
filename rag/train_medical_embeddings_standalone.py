#!/usr/bin/env python3
"""
Standalone medical embedding training script with CUDA support.
This script handles its own dependencies to ensure CUDA PyTorch is used.
"""

import subprocess
import sys
import os

# First, ensure we have CUDA PyTorch
def setup_cuda_environment():
    """Set up CUDA environment before importing other modules."""
    print("Setting up CUDA environment...")
    
    # Check if we already have CUDA torch
    try:
        import torch
        if torch.cuda.is_available() and "+cu" in torch.__version__:
            print(f"CUDA PyTorch already available: {torch.__version__}")
            return torch
    except ImportError:
        pass
    
    print("Installing CUDA PyTorch...")
    try:
        # Install CUDA PyTorch directly
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch==2.5.1+cu121", "torchvision==0.20.1+cu121", "torchaudio==2.5.1+cu121",
            "--index-url", "https://download.pytorch.org/whl/cu121",
            "--force-reinstall", "--no-deps"
        ], stdout=subprocess.DEVNULL)
        
        # Install necessary dependencies
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "sympy==1.13.1", "mpmath", "filelock", "fsspec", "jinja2", 
            "markupsafe", "networkx", "numpy", "pillow", "typing-extensions"
        ], stdout=subprocess.DEVNULL)
        
        # Reload torch
        if 'torch' in sys.modules:
            del sys.modules['torch']
        
        import torch
        return torch
        
    except Exception as e:
        print(f"Failed to install CUDA PyTorch: {e}")
        import torch  # Fallback
        return torch

# Set up CUDA before other imports
torch = setup_cuda_environment()

# Now import the rest
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Install and import sentence-transformers
try:
    from sentence_transformers import (
        SentenceTransformer, SentenceTransformerModelCardData, 
        SentenceTransformerTrainingArguments, SentenceTransformerTrainer
    )
    from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
    from sentence_transformers.util import cos_sim
    from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers
except ImportError:
    print("Installing sentence-transformers...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "sentence-transformers>=2.2.0"
    ])
    from sentence_transformers import (
        SentenceTransformer, SentenceTransformerModelCardData, 
        SentenceTransformerTrainingArguments, SentenceTransformerTrainer
    )
    from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
    from sentence_transformers.util import cos_sim
    from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

# Install and import datasets
try:
    from datasets import Dataset
except ImportError:
    print("Installing datasets...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets>=2.0.0"])
    from datasets import Dataset

class MedicalEmbeddingTrainer:
    """Standalone medical embedding trainer with CUDA support."""
    
    def __init__(
        self,
        model_id: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        dataset_path: str = "medical_embedding_dataset.json",
        output_dir: str = None,
        checkpoint_dir: str = None,
        matryoshka_dimensions: List[int] = None,
        device: str = None
    ):
        self.model_id = model_id
        self.dataset_path = dataset_path
        self.matryoshka_dimensions = matryoshka_dimensions or [768, 512, 256, 128, 64]
        
        # Create timestamped model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"biobert-medical-embeddings-mrl_{timestamp}"
        
        # Set up directories
        self.models_dir = Path("models")
        self.checkpoints_dir = Path("checkpoints")
        self.models_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        self.output_dir = output_dir or str(self.models_dir / model_name)
        self.checkpoint_dir = checkpoint_dir or str(self.checkpoints_dir / f"{model_name}_checkpoints")
        
        # Enhanced CUDA device detection
        self.device = self._setup_device(device)
        self._validate_cuda_setup()
        
        # Initialize placeholders
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.corpus = None
        self.queries = None
        self.relevant_docs = None
        
        logger.info(f"Final model will be saved to: {self.output_dir}")
        logger.info(f"Training checkpoints will be saved to: {self.checkpoint_dir}")
    
    def _setup_device(self, device: str = None) -> str:
        """Enhanced device setup with CUDA validation."""
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
                logger.warning("CUDA not available, falling back to CPU")
        else:
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                device = "cpu"
        
        return device
    
    def _validate_cuda_setup(self):
        """Validate and report CUDA setup."""
        if self.device == "cuda":
            try:
                # Test CUDA functionality
                test_tensor = torch.rand(2, 2).cuda()
                _ = test_tensor + test_tensor
                
                logger.info("CUDA validation successful!")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
                logger.info(f"CUDA version: {torch.version.cuda}")
                logger.info(f"PyTorch version: {torch.__version__}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                logger.info(f"Initial GPU memory allocated: {memory_allocated:.2f} GB")
                
            except Exception as e:
                logger.error(f"CUDA validation failed: {e}")
                logger.warning("Falling back to CPU")
                self.device = "cpu"
        
        if self.device == "cpu":
            logger.info("Using CPU device for training")
            logger.warning("Training will be significantly slower on CPU")
    
    def load_and_prepare_data(self, test_split: float = 0.1) -> None:
        """Load medical dataset and prepare for training."""
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = data['examples']
        logger.info(f"Loaded {len(examples)} triplet examples")
        
        # Split and convert to positive pairs
        split_idx = int(len(examples) * (1 - test_split))
        train_triplets = examples[:split_idx]
        eval_triplets = examples[split_idx:]
        
        train_examples = [{'anchor': ex['anchor'], 'positive': ex['positive']} 
                         for ex in train_triplets]
        eval_examples = [{'anchor': ex['anchor'], 'positive': ex['positive']} 
                        for ex in eval_triplets]
        
        self.train_dataset = Dataset.from_list(train_examples)
        self.eval_dataset = Dataset.from_list(eval_examples)
        
        self.train_dataset = self.train_dataset.add_column("id", range(len(self.train_dataset)))
        self.eval_dataset = self.eval_dataset.add_column("id", range(len(self.eval_dataset)))
        
        logger.info(f"Created train dataset: {len(self.train_dataset)} examples")
        logger.info(f"Created eval dataset: {len(self.eval_dataset)} examples")
        
        self._prepare_evaluation_data()
    
    def _prepare_evaluation_data(self) -> None:
        """Prepare evaluation data structures."""
        corpus_texts = list(set(self.eval_dataset['positive']))
        self.corpus = {i: text for i, text in enumerate(corpus_texts)}
        self.queries = {i: anchor for i, anchor in enumerate(self.eval_dataset['anchor'])}
        
        self.relevant_docs = {}
        for query_id, (anchor, positive) in enumerate(zip(self.eval_dataset['anchor'], 
                                                         self.eval_dataset['positive'])):
            matching_corpus_ids = [
                corpus_id for corpus_id, corpus_text in self.corpus.items()
                if corpus_text == positive
            ]
            self.relevant_docs[query_id] = matching_corpus_ids
        
        logger.info(f"Created corpus with {len(self.corpus)} documents")
        logger.info(f"Created {len(self.queries)} queries")
    
    def load_model(self) -> None:
        """Load BioBERT model for fine-tuning."""
        logger.info(f"Loading model: {self.model_id}")
        
        model_kwargs = {}
        if self.device == "cuda":
            model_kwargs["attn_implementation"] = "sdpa"
        
        self.model = SentenceTransformer(
            self.model_id,
            device=self.device,
            model_kwargs=model_kwargs,
            model_card_data=SentenceTransformerModelCardData(
                language="en",
                license="apache-2.0",
                model_name="BioBERT Medical Domain Embeddings with Matryoshka",
            ),
        )
        
        if self.device == "cuda":
            self.model = self.model.to(self.device)
            self.model.train()
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"Model loaded on CUDA. GPU memory allocated: {memory_allocated:.2f} GB")
        else:
            logger.info("Model loaded on CPU")
    
    def create_evaluator(self) -> SequentialEvaluator:
        """Create evaluation pipeline."""
        evaluators = []
        for dim in self.matryoshka_dimensions:
            ir_evaluator = InformationRetrievalEvaluator(
                queries=self.queries,
                corpus=self.corpus,
                relevant_docs=self.relevant_docs,
                name=f"dim_{dim}",
                truncate_dim=dim,
                score_functions={"cosine": cos_sim},
            )
            evaluators.append(ir_evaluator)
        return SequentialEvaluator(evaluators)
    
    def train(self) -> None:
        """Fine-tune the model."""
        logger.info("Starting fine-tuning...")
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Create loss functions
        base_loss = MultipleNegativesRankingLoss(self.model)
        train_loss = MatryoshkaLoss(
            self.model, base_loss, matryoshka_dims=self.matryoshka_dimensions
        )
        
        # Create evaluator
        evaluator = self.create_evaluator()
        
        # Optimize batch sizes for your RTX 3070 Ti
        if self.device == "cuda":
            train_batch_size = 8
            eval_batch_size = 4
            gradient_accumulation_steps = 64
            optimizer = "adamw_torch_fused"
            use_tf32 = True
            use_bf16 = True
        else:
            train_batch_size = 16
            eval_batch_size = 8
            gradient_accumulation_steps = 32
            optimizer = "adamw_torch"
            use_tf32 = False
            use_bf16 = False
        
        # Training arguments
        args = SentenceTransformerTrainingArguments(
            output_dir=self.checkpoint_dir,
            num_train_epochs=4,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=eval_batch_size,
            warmup_ratio=0.1,
            learning_rate=2e-5,
            lr_scheduler_type="cosine",
            optim=optimizer,
            tf32=use_tf32,
            bf16=use_bf16,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_dim_256_cosine_ndcg@10",
            greater_is_better=True,
            report_to="none",
            dataloader_drop_last=False,
            remove_unused_columns=True,
        )
        
        # Create trainer
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset.select_columns(["positive", "anchor"]),
            loss=train_loss,
            evaluator=evaluator,
        )
        
        # Train
        logger.info("Training started...")
        if self.device == "cuda":
            memory_before = torch.cuda.memory_allocated() / 1e9
            logger.info(f"GPU memory before training: {memory_before:.2f} GB")
        
        trainer.train()
        
        if self.device == "cuda":
            memory_after = torch.cuda.memory_allocated() / 1e9
            logger.info(f"GPU memory after training: {memory_after:.2f} GB")
            torch.cuda.empty_cache()
        
        # Save final model
        logger.info(f"Saving final model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        
        # Save model info
        model_info = {
            'model_id': self.model_id,
            'training_timestamp': datetime.now().isoformat(),
            'dataset_path': self.dataset_path,
            'matryoshka_dimensions': self.matryoshka_dimensions,
            'device': self.device,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'training_epochs': 4
        }
        
        info_path = os.path.join(self.output_dir, 'model_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Final model saved to {self.output_dir}")
        logger.info(f"Training checkpoints saved to {self.checkpoint_dir}")

def main():
    """Main training function."""
    # Check for dataset
    dataset_path = "improved_medical_embedding_dataset.json"
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        logger.info("Please ensure the medical dataset is available")
        return False
    
    # Check NVIDIA GPU
    try:
        result = subprocess.run("nvidia-smi", capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("NVIDIA GPU detected")
        else:
            logger.warning("NVIDIA GPU not detected")
    except FileNotFoundError:
        logger.warning("nvidia-smi not found")
    
    # Initialize trainer
    trainer = MedicalEmbeddingTrainer()
    
    # Log device info
    if trainer.device == "cuda":
        logger.info("CUDA acceleration enabled!")
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("Training will run on CPU")
    
    try:
        # Load data and model
        trainer.load_and_prepare_data()
        trainer.load_model()
        
        # Train
        trainer.train()
        
        logger.info("Training completed successfully!")
        print(f"\n{'='*80}")
        print("TRAINING COMPLETED!")
        print(f"{'='*80}")
        print(f"Model saved to: {trainer.output_dir}")
        print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
        print(f"Device used: {trainer.device.upper()}")
        print("\nTo use the model:")
        print(f"from sentence_transformers import SentenceTransformer")
        print(f"model = SentenceTransformer('{trainer.output_dir}', truncate_dim=256)")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)