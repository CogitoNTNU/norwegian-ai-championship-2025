"""
Fine-tuning BioBERT embedding model for medical domain RAG system.
Based on the FT_Embedding_Models_on_Domain_Specific_Data.ipynb notebook,
adapted for the medical domain.
"""

import json
import os
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Configure logging at the top
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import torch after ensuring CUDA version
def ensure_cuda_torch():
    """Ensure CUDA-enabled PyTorch is installed."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch  # CUDA PyTorch is already available
        else:
            logger.warning("PyTorch CUDA not available, attempting to install...")
    except ImportError:
        logger.warning("PyTorch not found, installing CUDA version...")
    
    # Install CUDA PyTorch
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch==2.5.1+cu121", "torchvision==0.20.1+cu121", "torchaudio==2.5.1+cu121",
            "--index-url", "https://download.pytorch.org/whl/cu121", "--no-deps"
        ], stdout=subprocess.DEVNULL)
        
        # Install dependencies
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "sympy", "mpmath", "filelock", "fsspec", "jinja2", "markupsafe", 
            "networkx", "numpy", "pillow", "typing-extensions"
        ], stdout=subprocess.DEVNULL)
        
        # Reimport torch
        import importlib
        if 'torch' in sys.modules:
            importlib.reload(sys.modules['torch'])
        import torch
        
        return torch
        
    except Exception as e:
        logger.error(f"Failed to install CUDA PyTorch: {e}")
        import torch  # Fall back to whatever is available
        return torch

# Ensure CUDA PyTorch before other imports
torch = ensure_cuda_torch()

# SentenceTransformers imports
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData, SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.util import cos_sim
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

# Datasets
from datasets import Dataset, load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class MedicalEmbeddingTrainer:
    """
    Fine-tuning trainer for medical domain embedding models using BioBERT.
    """
    
    def __init__(
        self,
        model_id: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        dataset_path: str = "medical_embedding_dataset.json",
        output_dir: str = None,
        checkpoint_dir: str = None,
        matryoshka_dimensions: List[int] = None,
        device: str = None
    ):
        """
        Initialize the medical embedding trainer.
        
        Args:
            model_id: HuggingFace model ID for BioBERT
            dataset_path: Path to the medical dataset JSON file
            output_dir: Output directory for the final trained model (defaults to models/)
            checkpoint_dir: Directory for training checkpoints (defaults to checkpoints/)
            matryoshka_dimensions: List of embedding dimensions for MRL
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
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
        
        # Enhanced CUDA device detection and validation
        self.device = self._setup_device(device)
        self._validate_cuda_setup()
        
        # Initialize model and datasets
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
            # Auto-detect best available device
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
                logger.warning("CUDA not available, falling back to CPU")
        else:
            # Validate requested device
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
                _ = test_tensor + test_tensor  # Simple operation test
                
                logger.info("✅ CUDA validation successful!")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
                logger.info(f"CUDA version: {torch.version.cuda}")
                logger.info(f"PyTorch version: {torch.__version__}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Memory check
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
        """
        Load the medical dataset and prepare it for training and evaluation.
        
        Args:
            test_split: Fraction of data to use for evaluation
        """
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        # Load the triplet dataset
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        examples = data['examples']
        logger.info(f"Loaded {len(examples)} triplet examples")
        
        # Convert triplets to positive pairs for training
        train_examples = []
        eval_examples = []
        
        # Split data into train/eval
        split_idx = int(len(examples) * (1 - test_split))
        train_triplets = examples[:split_idx]
        eval_triplets = examples[split_idx:]
        
        # Convert triplets to anchor-positive pairs
        for example in train_triplets:
            train_examples.append({
                'anchor': example['anchor'],
                'positive': example['positive']
            })
        
        for example in eval_triplets:
            eval_examples.append({
                'anchor': example['anchor'], 
                'positive': example['positive']
            })
        
        # Create datasets
        self.train_dataset = Dataset.from_list(train_examples)
        self.eval_dataset = Dataset.from_list(eval_examples)
        
        # Add IDs to datasets
        self.train_dataset = self.train_dataset.add_column("id", range(len(self.train_dataset)))
        self.eval_dataset = self.eval_dataset.add_column("id", range(len(self.eval_dataset)))
        
        logger.info(f"Created train dataset: {len(self.train_dataset)} examples")
        logger.info(f"Created eval dataset: {len(self.eval_dataset)} examples")
        
        # Prepare evaluation data structures
        self._prepare_evaluation_data()
        
    def _prepare_evaluation_data(self) -> None:
        """
        Prepare data structures for InformationRetrievalEvaluator.
        Creates corpus, queries, and relevant_docs mappings.
        """
        # Create corpus from all positive texts in eval dataset
        # Each positive text becomes a document in the corpus
        corpus_texts = list(set(self.eval_dataset['positive']))  # Remove duplicates
        
        self.corpus = {i: text for i, text in enumerate(corpus_texts)}
        
        # Create queries from eval dataset anchors
        self.queries = {i: anchor for i, anchor in enumerate(self.eval_dataset['anchor'])}
        
        # Create relevance mapping
        self.relevant_docs = {}
        
        for query_id, (anchor, positive) in enumerate(zip(self.eval_dataset['anchor'], self.eval_dataset['positive'])):
            # Find corpus IDs that match this positive text
            matching_corpus_ids = [
                corpus_id for corpus_id, corpus_text in self.corpus.items()
                if corpus_text == positive
            ]
            self.relevant_docs[query_id] = matching_corpus_ids
        
        logger.info(f"Created corpus with {len(self.corpus)} documents")
        logger.info(f"Created {len(self.queries)} queries")
        logger.info(f"Average relevant docs per query: {sum(len(docs) for docs in self.relevant_docs.values()) / len(self.relevant_docs):.2f}")
    
    def load_model(self) -> None:
        """Load the BioBERT model for fine-tuning with CUDA optimization."""
        logger.info(f"Loading model: {self.model_id}")
        
        # Load model with device specification
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
        
        # Move model to device explicitly if CUDA
        if self.device == "cuda":
            self.model = self.model.to(self.device)
            
            # Set model to training mode for fine-tuning
            self.model.train()
            
            # Log memory usage
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"Model loaded on CUDA. GPU memory allocated: {memory_allocated:.2f} GB")
        else:
            logger.info("Model loaded on CPU")
        
        logger.info(f"Model loaded successfully. Device: {self.model.device}")
        
    def create_evaluator(self) -> SequentialEvaluator:
        """
        Create the evaluation pipeline with multiple Matryoshka dimensions.
        
        Returns:
            SequentialEvaluator for multi-dimensional evaluation
        """
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
    
    def evaluate_base_model(self) -> Dict[str, float]:
        """
        Evaluate the base model performance before fine-tuning.
        
        Returns:
            Dictionary of evaluation results
        """
        logger.info("Evaluating base model performance...")
        
        evaluator = self.create_evaluator()
        base_results = evaluator(self.model)
        
        self._print_evaluation_results(base_results, "Base Model")
        
        return base_results
    
    def train(self) -> None:
        """
        Fine-tune the BioBERT model on medical data with CUDA optimization.
        """
        logger.info("Starting fine-tuning...")
        
        # Clear CUDA cache before training
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared before training")
        
        # Create loss functions
        base_loss = MultipleNegativesRankingLoss(self.model)
        train_loss = MatryoshkaLoss(
            self.model, base_loss, matryoshka_dims=self.matryoshka_dimensions
        )
        
        # Create evaluator
        evaluator = self.create_evaluator()
        
        # CUDA-specific batch size adjustments
        if self.device == "cuda":
            # Optimize batch sizes for RTX 3070 Ti (8GB VRAM)
            train_batch_size = 8  # Smaller for better memory usage
            eval_batch_size = 4
            gradient_accumulation_steps = 64  # Increase to maintain effective batch size
            optimizer = "adamw_torch_fused"  # CUDA-optimized optimizer
            use_tf32 = True
            use_bf16 = True
        else:
            train_batch_size = 16
            eval_batch_size = 8
            gradient_accumulation_steps = 32
            optimizer = "adamw_torch"
            use_tf32 = False
            use_bf16 = False
        
        # Training arguments optimized for medical domain and device
        args = SentenceTransformerTrainingArguments(
            output_dir=self.checkpoint_dir,  # Use checkpoint directory for training output
            num_train_epochs=4,  # Reduced for testing
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=eval_batch_size,
            warmup_ratio=0.1,
            learning_rate=2e-5,  # Conservative learning rate for domain adaptation
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
            metric_for_best_model="eval_dim_256_cosine_ndcg@10",  # Optimize for 256d NDCG@10
            greater_is_better=True,
            report_to="none",  # Set to "wandb" if you want W&B logging
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
        
        # Start training with memory monitoring
        logger.info("Training started...")
        if self.device == "cuda":
            memory_before = torch.cuda.memory_allocated() / 1e9
            logger.info(f"GPU memory before training: {memory_before:.2f} GB")
        
        trainer.train()
        
        # Monitor memory after training
        if self.device == "cuda":
            memory_after = torch.cuda.memory_allocated() / 1e9
            logger.info(f"GPU memory after training: {memory_after:.2f} GB")
            torch.cuda.empty_cache()  # Clean up after training
        
        # Save the best model to final location
        logger.info(f"Saving final model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        
        # Create a model info file
        model_info = {
            'model_id': self.model_id,
            'training_timestamp': datetime.now().isoformat(),
            'dataset_path': self.dataset_path,
            'matryoshka_dimensions': self.matryoshka_dimensions,
            'device': self.device,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'training_epochs': 4
        }
        
        info_path = os.path.join(self.output_dir, 'model_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Final model saved to {self.output_dir}")
        logger.info(f"Training checkpoints saved to {self.checkpoint_dir}")
        
        return trainer
    
    def evaluate_final_model(self) -> Dict[str, float]:
        """
        Evaluate the fine-tuned model performance with CUDA support.
        
        Returns:
            Dictionary of evaluation results
        """
        logger.info("Evaluating fine-tuned model...")
        
        # Load the fine-tuned model with device specification
        fine_tuned_model = SentenceTransformer(
            self.output_dir, 
            device=self.device
        )
        
        if self.device == "cuda":
            fine_tuned_model = fine_tuned_model.to(self.device)
            logger.info("Fine-tuned model loaded on CUDA for evaluation")
        
        evaluator = self.create_evaluator()
        ft_results = evaluator(fine_tuned_model)
        
        self._print_evaluation_results(ft_results, "Fine-tuned Model")
        
        return ft_results
    
    def _print_evaluation_results(self, results: Dict[str, float], model_name: str) -> None:
        """
        Print evaluation results in a formatted table.
        
        Args:
            results: Dictionary of evaluation results
            model_name: Name to display in the table header
        """
        print(f"\n{model_name} Evaluation Results")
        print("-" * 85)
        print(f"{'Metric':15} {'768d':>12} {'512d':>12} {'256d':>12} {'128d':>12} {'64d':>12}")
        print("-" * 85)
        
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
        
        for metric in metrics:
            values = []
            for dim in self.matryoshka_dimensions:
                key = f"dim_{dim}_cosine_{metric}"
                values.append(results.get(key, 0.0))
            
            # Highlight NDCG@10
            metric_name = f"=={metric}==" if metric == "ndcg@10" else metric
            print(f"{metric_name:15}", end="  ")
            for val in values:
                print(f"{val:12.4f}", end=" ")
            print()
        
        print("-" * 85)
        print(f"{'seq_score:'} {results.get('sequential_score', 0.0):.4f}")
    
    def compare_models(self, base_results: Dict[str, float], ft_results: Dict[str, float]) -> None:
        """
        Compare base and fine-tuned model performance.
        
        Args:
            base_results: Base model evaluation results
            ft_results: Fine-tuned model evaluation results
        """
        print("\nBase vs Fine-tuned Model Comparison")
        print("-" * 100)
        print(f"{'Metric':15} {'Dim':>5} {'Base':>12} {'Fine-tuned':>12} {'Abs. Imp.':>12} {'% Imp.':>10}")
        print("-" * 100)
        
        key_metrics = ['ndcg@10', 'mrr@10', 'map@100', 'accuracy@10']
        
        for metric in key_metrics:
            for dim in self.matryoshka_dimensions:
                base_key = f"dim_{dim}_cosine_{metric}"
                base_val = base_results.get(base_key, 0.0)
                ft_val = ft_results.get(base_key, 0.0)
                
                abs_improvement = ft_val - base_val
                pct_improvement = (abs_improvement / base_val * 100) if base_val > 0 else 0
                
                print(f"{metric:15} {dim:>5}d {base_val:12.4f} {ft_val:12.4f} {abs_improvement:12.4f} {pct_improvement:9.1f}%")
    
    def run_full_pipeline(self) -> None:
        """
        Run the complete training and evaluation pipeline.
        """
        logger.info("Starting full medical embedding training pipeline...")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Load model
        self.load_model()
        
        # Evaluate base model
        base_results = self.evaluate_base_model()
        
        # Train model
        self.train()
        
        # Evaluate fine-tuned model
        ft_results = self.evaluate_final_model()
        
        # Compare results
        self.compare_models(base_results, ft_results)
        
        logger.info("Training pipeline completed successfully!")
        
        # Save comparison results
        comparison_results = {
            'base_model_results': base_results,
            'fine_tuned_results': ft_results,
            'model_id': self.model_id,
            'dataset_path': self.dataset_path,
            'output_dir': self.output_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'matryoshka_dimensions': self.matryoshka_dimensions
        }
        
        results_path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Evaluation results saved to {results_path}")


def main():
    """
    Main function to run the medical embedding training with CUDA support.
    """
    # Configuration
    config = {
        "model_id": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        "dataset_path": "improved_medical_embedding_dataset.json",
        "output_dir": None,  # Will auto-generate in models/ folder
        "checkpoint_dir": None,  # Will auto-generate in checkpoints/ folder
        "matryoshka_dimensions": [768, 512, 256, 128, 64],
        "device": None  # Auto-detect CUDA or specify "cuda"/"cpu"
    }
    
    # Check if dataset exists
    if not os.path.exists(config["dataset_path"]):
        logger.error(f"Dataset not found at {config['dataset_path']}")
        logger.info("Please run fine_tune_embeddings_model.py first to generate the dataset")
        return
    
    # Initialize trainer with CUDA support
    trainer = MedicalEmbeddingTrainer(**config)
    
    # Log device info
    if trainer.device == "cuda":
        logger.info("🚀 CUDA acceleration enabled for training!")
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("Training will run on CPU")
    
    # Run the full pipeline
    try:
        trainer.run_full_pipeline()
        logger.info("Medical embedding training completed successfully!")
        
        # Print usage instructions
        print("\n" + "="*80)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nYour fine-tuned BioBERT model is saved at: {trainer.output_dir}")
        print(f"Training checkpoints are saved at: {trainer.checkpoint_dir}")
        print(f"Original BioBERT model: {config['model_id']}")
        print(f"Training device: {trainer.device.upper()}")
        print("\nTo use your fine-tuned model:")
        print("```python")
        print("from sentence_transformers import SentenceTransformer")
        print(f"model = SentenceTransformer('{trainer.output_dir}', truncate_dim=256)")
        if trainer.device == "cuda":
            print("# Model will automatically use CUDA if available")
        print("embeddings = model.encode(['your medical text here'])")
        print("```")
        print("\nFor different Matryoshka dimensions, use truncate_dim parameter:")
        print("- truncate_dim=768  # Full dimension")
        print("- truncate_dim=256  # Balanced performance/efficiency") 
        print("- truncate_dim=128  # Fast retrieval")
        print("- truncate_dim=64   # Ultra-fast retrieval")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()