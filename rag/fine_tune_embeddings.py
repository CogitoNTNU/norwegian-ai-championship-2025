import subprocess, sys, os, importlib.util, pkg_resources

def _need_upgrade() -> bool:
    spec = importlib.util.find_spec("sentence_transformers")
    if not spec:
        return True
    ver = pkg_resources.get_distribution("sentence-transformers").version
    major = int(ver.split(".")[0])
    try:
        import huggingface_hub as hub
        ok_hub = hasattr(hub, "hf_hub_download")
    except ImportError:
        ok_hub = False
    return major < 3 or not ok_hub

if _need_upgrade():
    print("[bootstrap] Installing/upgrading packages …", flush=True)
    cmd = [
        sys.executable, "-m", "uv", "pip", "install",
        # ----- explicit indexes -----
        "--index-url", "https://pypi.org/simple",
        "--extra-index-url", "https://download.pytorch.org/whl/cu121",
        # ----- required wheels -----
        "sentence-transformers==2.7.0",      # modern nok, finnes på PyPI
        "huggingface_hub==0.25.2",           # matcher 2.7.0
        "torch==2.3.0",                      # CUDA-12.1-wheel
    ]
    subprocess.check_call(cmd)
    os.execv(sys.executable, [sys.executable] + sys.argv)#!/usr/bin/env python
"""Fine-tune embedding models for medical statement classification."""

import subprocess, sys, os

def _require_modern_sbert():
    try:
        import sentence_transformers as st, huggingface_hub as hub
        if tuple(map(int, st.__version__.split('.')[:2])) >= (3, 0) \
           and hasattr(hub, "hf_hub_download"):
            return
        raise ImportError
    except Exception:
        print("[bootstrap] installing or upgrading packages …", flush=True)
        subprocess.check_call([
            "uv", "pip", "install", "-qq",
            "sentence-transformers>=4.0.0,<6.0.0",
            "huggingface_hub>=0.28.0",
            "torch==2.3.0",
            "--index-url", "https://download.pytorch.org/whl/cu121",
        ])
        os.execv(sys.executable, [sys.executable] + sys.argv)

_require_modern_sbert()


import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from datetime import datetime
import sys
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import InputExample


# Add rag-pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag-pipeline"))

from document_store_embeddings import EmbeddingsDocumentStore


class MedicalTripletDataset(Dataset):
    """Dataset for medical statement-document triplets."""
    
    def __init__(
        self,
        statements: List[str],
        statements_topics: List[int],
        document_chunks: List[Dict[str, Any]],
        topic_mapping: Dict[int, str],
        negative_ratio: int = 2
    ):
        """
        Initialize dataset with statements and document chunks.
        
        Args:
            statements: List of statement texts
            statements_topics: List of topic IDs for statements
            document_chunks: List of document chunks with metadata
            topic_mapping: Mapping from topic ID to topic name
            negative_ratio: Number of negatives per positive pair
        """
        self.statements = statements
        self.statements_topics = statements_topics
        self.document_chunks = document_chunks
        self.topic_mapping = topic_mapping
        self.negative_ratio = negative_ratio
        
        # Group chunks by topic for efficient sampling
        self.chunks_by_topic = {}
        for chunk in document_chunks:
            topic_id = chunk['metadata']['topic_id']
            if topic_id not in self.chunks_by_topic:
                self.chunks_by_topic[topic_id] = []
            self.chunks_by_topic[topic_id].append(chunk)
        
        # Create training examples
        self.examples = self._create_training_examples()
    
    def _create_training_examples(self) -> List[InputExample]:
        """Create training examples with positive and negative pairs."""
        examples = []
        
        for i, (statement, topic) in enumerate(zip(self.statements, self.statements_topics)):
            # Get positive chunks (same topic)
            if topic in self.chunks_by_topic:
                positive_chunks = self.chunks_by_topic[topic]
                
                # Sample positive examples
                for pos_chunk in random.sample(
                    positive_chunks, 
                    min(3, len(positive_chunks))  # Max 3 positives per statement
                ):
                    examples.append(InputExample(
                        texts=[statement, pos_chunk['chunk']],
                        label=1.0
                    ))
                
                # Sample hard negatives (related medical topics)
                negative_topics = self._get_related_topics(topic)
                for neg_topic in negative_topics[:self.negative_ratio]:
                    if neg_topic in self.chunks_by_topic:
                        neg_chunks = self.chunks_by_topic[neg_topic]
                        neg_chunk = random.choice(neg_chunks)
                        examples.append(InputExample(
                            texts=[statement, neg_chunk['chunk']],
                            label=0.0
                        ))
        
        return examples
    
    def _get_related_topics(self, target_topic: int) -> List[int]:
        """Get related medical topics for hard negative sampling."""
        # Medical topic clustering - these are medically related topics
        medical_clusters = {
            # Cardiac conditions
            frozenset([4, 7, 22, 23, 24, 25, 38, 49, 51, 57, 69, 77, 80, 82]),
            # Respiratory conditions  
            frozenset([8, 13, 14, 19, 21, 34, 45, 46, 47, 59, 61, 62, 63, 64, 65, 66, 67, 74, 81]),
            # Trauma conditions
            frozenset([0, 16, 20, 26, 28, 39, 55, 79]),
            # GI conditions
            frozenset([1, 2, 3, 17, 37, 54, 56]),
            # Neurological conditions  
            frozenset([18, 29, 35, 48, 71, 75, 76]),
            # Infections/Sepsis
            frozenset([36, 72, 61, 48, 35]),
            # Metabolic/Endocrine
            frozenset([30, 42, 43, 44, 6, 5]),
            # Reproductive/Gynecological
            frozenset([31, 32, 52, 58, 78]),
            # Tests/Procedures
            frozenset(range(83, 117))  # All test/procedure topics
        }
        
        # Find cluster containing target topic
        target_cluster = None
        for cluster in medical_clusters:
            if target_topic in cluster:
                target_cluster = cluster
                break
        
        if target_cluster:
            # Return other topics in same cluster (excluding target)
            related = list(target_cluster - {target_topic})
            return random.sample(related, min(5, len(related)))
        else:
            # Return random topics if not in any cluster
            all_topics = list(self.chunks_by_topic.keys())
            available = [t for t in all_topics if t != target_topic]
            return random.sample(available, min(5, len(available)))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class EmbeddingFineTuner:
    """Fine-tune embedding models for medical statement classification."""
    
    def __init__(
        self,
        base_model: str = "NeuML/pubmedbert-base-embeddings",
        device: str = None
    ):
        """
        Initialize fine-tuner.
        
        Args:
            base_model: Base embedding model to fine-tune
            device: Device to use (cuda/cpu/auto)
        """
        self.base_model = base_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load base model
        print(f"Loading base model: {base_model}")
        self.model = SentenceTransformer(base_model, device=self.device)
        
    def prepare_training_data(
        self,
        statements_dir: str,
        answers_dir: str,
        topics_dir: str,
        topics_json: str,
        max_statements: int = None
    ) -> MedicalTripletDataset:
        """
        Prepare training data from statement files and medical documents.
        
        Args:
            statements_dir: Directory with statement text files
            answers_dir: Directory with answer JSON files
            topics_dir: Directory with medical topic documents
            topics_json: Path to topics mapping JSON
            max_statements: Maximum statements to use (None for all)
            
        Returns:
            MedicalTripletDataset for training
        """
        print("Preparing training data...")
        
        # Load statements and answers
        statements = []
        topics = []
        
        statements_path = Path(statements_dir)
        answers_path = Path(answers_dir)
        
        statement_files = sorted(list(statements_path.glob("statement_*.txt")))
        if max_statements:
            statement_files = statement_files[:max_statements]
        
        for statement_file in statement_files:
            statement_id = statement_file.stem
            answer_file = answers_path / f"{statement_id}.json"
            
            if not answer_file.exists():
                continue
            
            # Load statement
            with open(statement_file, 'r') as f:
                statement = f.read().strip()
            
            # Load answer
            with open(answer_file, 'r') as f:
                answer = json.load(f)
            
            statements.append(statement)
            topics.append(answer['statement_topic'])
        
        print(f"Loaded {len(statements)} statements")
        
        # Load medical documents using document store
        doc_store = EmbeddingsDocumentStore("pubmedbert-base-embeddings", device=self.device)
        doc_store.load_medical_documents(topics_dir, topics_json)
        
        # Get document chunks
        document_chunks = []
        for i, chunk_text in enumerate(doc_store.chunks):
            chunk_metadata = doc_store.chunk_metadata[i]
            document_chunks.append({
                'chunk': chunk_text,
                'metadata': chunk_metadata
            })
        
        print(f"Loaded {len(document_chunks)} document chunks")
        
        # Load topic mapping
        with open(topics_json, 'r') as f:
            topic_mapping = json.load(f)
        
        # Create dataset
        dataset = MedicalTripletDataset(
            statements=statements,
            statements_topics=topics,
            document_chunks=document_chunks,
            topic_mapping=topic_mapping
        )
        
        print(f"Created dataset with {len(dataset)} training examples")
        return dataset
    
    def fine_tune(
        self,
        dataset: MedicalTripletDataset,
        output_path: str,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100,
        evaluation_steps: int = 500
    ) -> str:
        """
        Fine-tune the embedding model.
        
        Args:
            dataset: Training dataset
            output_path: Path to save fine-tuned model
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Warmup steps for learning rate
            evaluation_steps: Steps between evaluations
            
        Returns:
            Path to saved model
        """
        print(f"Starting fine-tuning on {self.device}...")
        
        # Create data loader
        train_dataloader = DataLoader(
            dataset, 
            shuffle=True, 
            batch_size=batch_size
        )
        
        # Define loss function
        train_loss = MultipleNegativesRankingLoss(model=self.model)
        
        # Create evaluation data (use subset for speed)
        eval_examples = dataset.examples[:min(100, len(dataset.examples))]
        queries = {}
        corpus = {}
        relevant_docs = {}
        
        for i, example in enumerate(eval_examples):
            query_id = f"q{i}"
            doc_id = f"d{i}"
            
            queries[query_id] = example.texts[0]
            corpus[doc_id] = example.texts[1]
            relevant_docs[query_id] = {doc_id: 1}
        
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="medical_eval"
        )
        
        # Fine-tune model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            evaluator=evaluator,
            evaluation_steps=evaluation_steps,
            save_best_model=True,
            optimizer_params={'lr': learning_rate}
        )
        
        print(f"Fine-tuning completed. Model saved to: {output_path}")
        return output_path
    
    def evaluate_on_retrieval(
        self,
        model_path: str,
        statements_dir: str,
        answers_dir: str,
        topics_dir: str,
        topics_json: str,
        max_samples: int = 50
    ) -> Dict[str, float]:
        """
        Evaluate fine-tuned model on retrieval task.
        
        Args:
            model_path: Path to fine-tuned model
            statements_dir: Directory with test statements
            answers_dir: Directory with test answers
            topics_dir: Directory with medical documents
            topics_json: Path to topics mapping
            max_samples: Number of samples to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating fine-tuned model...")
        
        # Load fine-tuned model
        finetuned_model = SentenceTransformer(model_path, device=self.device)
        
        # Setup document store with fine-tuned model
        doc_store = EmbeddingsDocumentStore("custom", device=self.device)
        doc_store.model = finetuned_model  # Replace with fine-tuned model
        doc_store.load_medical_documents(topics_dir, topics_json)
        doc_store.build_index()
        
        # Load test statements
        statements_path = Path(statements_dir)
        answers_path = Path(answers_dir)
        
        statement_files = sorted(list(statements_path.glob("statement_*.txt")))[
            :max_samples
        ]
        
        correct_retrievals = 0
        total_retrievals = 0
        mrr_scores = []
        
        for statement_file in statement_files:
            statement_id = statement_file.stem
            answer_file = answers_path / f"{statement_id}.json"
            
            if not answer_file.exists():
                continue
            
            # Load statement and answer
            with open(statement_file, 'r') as f:
                statement = f.read().strip()
            
            with open(answer_file, 'r') as f:
                answer = json.load(f)
            
            true_topic = answer['statement_topic']
            
            # Retrieve documents
            results = doc_store.similarity_search(statement, k=10)
            
            # Check if correct topic is in top results
            retrieved_topics = [r['metadata']['topic_id'] for r in results]
            
            if true_topic in retrieved_topics:
                correct_retrievals += 1
                # Calculate MRR
                rank = retrieved_topics.index(true_topic) + 1
                mrr_scores.append(1.0 / rank)
            else:
                mrr_scores.append(0.0)
            
            total_retrievals += 1
        
        # Calculate metrics
        retrieval_accuracy = correct_retrievals / total_retrievals if total_retrievals > 0 else 0
        mrr = np.mean(mrr_scores) if mrr_scores else 0
        
        results = {
            'retrieval_accuracy@10': retrieval_accuracy,
            'mrr@10': mrr,
            'total_samples': total_retrievals
        }
        
        print(f"Retrieval Results:")
        print(f"  Accuracy@10: {retrieval_accuracy:.3f}")
        print(f"  MRR@10: {mrr:.3f}")
        print(f"  Samples: {total_retrievals}")
        
        return results


def main():
    """Main fine-tuning pipeline."""
    
    # Configuration
    config = {
        'base_model': 'NeuML/pubmedbert-base-embeddings',
        'statements_dir': 'data/processed/combined_train/statements',
        'answers_dir': 'data/processed/combined_train/answers', 
        'topics_dir': 'data/raw/topics',
        'topics_json': 'data/topics.json',
        'max_statements': 500,  # Use subset for faster training
        'output_dir': f'models/fine_tuned_embeddings_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'epochs': 3,
        'batch_size': 16,
        'learning_rate': 2e-5
    }
    
    print("=== Medical Embedding Fine-Tuning Pipeline ===")
    print(f"Configuration: {config}")
    
    # Initialize fine-tuner
    fine_tuner = EmbeddingFineTuner(
        base_model=config['base_model'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Prepare training data
    dataset = fine_tuner.prepare_training_data(
        statements_dir=config['statements_dir'],
        answers_dir=config['answers_dir'],
        topics_dir=config['topics_dir'],
        topics_json=config['topics_json'],
        max_statements=config['max_statements']
    )
    
    # Fine-tune model
    model_path = fine_tuner.fine_tune(
        dataset=dataset,
        output_path=config['output_dir'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )
    
    # Evaluate on retrieval task
    results = fine_tuner.evaluate_on_retrieval(
        model_path=model_path,
        statements_dir=config['statements_dir'],
        answers_dir=config['answers_dir'],
        topics_dir=config['topics_dir'],
        topics_json=config['topics_json'],
        max_samples=100
    )
    
    # Save results
    results_file = Path(config['output_dir']) / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'config': config,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nFine-tuning completed!")
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()