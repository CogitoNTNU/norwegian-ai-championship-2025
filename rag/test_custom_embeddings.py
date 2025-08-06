#!/usr/bin/env python3
"""
Test RAG pipeline with custom embedding models including fine-tuned models.
Allows specifying local model paths or HuggingFace model identifiers.
"""

import os
import platform
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Configure multiprocessing based on platform
if platform.system() == "Darwin":  # macOS
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "10"
else:  # Windows/Linux
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if "OMP_NUM_THREADS" not in os.environ:
        import multiprocessing
        os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())

# Set custom cache directory for sentence-transformers
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(
    os.path.dirname(__file__), ".cache", "sentence_transformers"
)

# Add rag-pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag-pipeline"))

from sentence_transformers import SentenceTransformer
import torch


class CustomEmbeddingRAGPipeline:
    """RAG Pipeline with custom embedding model support."""
    
    def __init__(self, embedding_model_path: str, llm_model: str = "cogito:8b", 
                 device: str = "auto", top_k: int = 5):
        self.embedding_model_path = embedding_model_path
        self.llm_model = llm_model
        self.device = self._determine_device(device)
        self.top_k = top_k
        
        print(f"[INIT] Loading embedding model from: {embedding_model_path}")
        print(f"[INIT] Using device: {self.device}")
        
        # Load the custom embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model_path, device=self.device)
            model_info = f"Embedding dimensions: {self.embedding_model.get_sentence_embedding_dimension()}"
            print(f"[INIT] Successfully loaded model - {model_info}")
        except Exception as e:
            print(f"[ERROR] Failed to load embedding model: {e}")
            raise
            
        # Initialize other components
        self.documents = []
        self.embeddings = None
        
    def _determine_device(self, device: str) -> str:
        """Determine the best device to use."""
        cuda_available = torch.cuda.is_available()
        
        if device == "auto":
            if platform.system() == "Darwin":  # macOS
                return "cpu"  # Avoid MPS issues
            else:
                return "cuda" if cuda_available else "cpu"
        elif device == "cuda":
            if cuda_available:
                return "cuda"
            else:
                print("[WARNING] CUDA requested but not available, using CPU")
                return "cpu"
        else:
            return "cpu"
    
    def load_documents(self, chunks_file: str):
        """Load documents from chunks file."""
        print(f"[DATA] Loading documents from: {chunks_file}")
        
        if not os.path.exists(chunks_file):
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                self.documents.append(json.loads(line))
        
        print(f"[DATA] Loaded {len(self.documents)} documents")
        
        # Generate embeddings for all documents
        print("[DATA] Generating document embeddings...")
        document_texts = [doc["text"] for doc in self.documents]
        
        start_time = time.time()
        self.embeddings = self.embedding_model.encode(
            document_texts, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )
        encoding_time = time.time() - start_time
        
        print(f"[DATA] Generated embeddings in {encoding_time:.2f}s")
        print(f"[DATA] Embedding shape: {self.embeddings.shape}")
    
    def retrieve_documents(self, query: str) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        if self.embeddings is None:
            raise ValueError("Documents not loaded. Call load_documents() first.")
        
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Calculate similarities (cosine similarity)
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        
        retrieved_docs = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc["similarity_score"] = similarities[idx]
            retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def classify_statement(self, statement: str, retrieved_docs: List[Dict]) -> Tuple[bool, int]:
        """Classify a statement using retrieved context."""
        # Create context from retrieved documents
        context = "\n\n".join([doc["text"][:500] + "..." if len(doc["text"]) > 500 else doc["text"] 
                              for doc in retrieved_docs])
        
        # Simple rule-based classification for demonstration
        # In practice, you'd use your LLM here
        
        # For binary classification: look for contradictory or supporting keywords
        statement_lower = statement.lower()
        context_lower = context.lower()
        
        # Simple heuristic - in real implementation, use LLM
        binary_prediction = True  # Default to True
        topic_prediction = 1      # Default topic
        
        # Try to extract some signal from the context
        if any(word in context_lower for word in ["not", "never", "no", "false", "incorrect"]):
            if any(word in statement_lower for word in ["not", "never", "no"]):
                binary_prediction = True  # Double negative
            else:
                binary_prediction = False
        
        return binary_prediction, topic_prediction
    
    def predict(self, statement: str) -> Tuple[bool, int]:
        """Make a prediction for a statement."""
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(statement)
        
        # Classify based on retrieved context
        binary_pred, topic_pred = self.classify_statement(statement, retrieved_docs)
        
        return binary_pred, topic_pred


def load_training_statements(n: int) -> List[Tuple[str, Dict]]:
    """Load n statements from training data."""
    data_dir = Path(__file__).parent / "data"
    
    # Direct path to the known structure
    statements_dir = data_dir / "processed" / "combined" / "statements"
    answers_dir = data_dir / "processed" / "combined" / "answers"
    
    if not statements_dir.exists():
        print(f"[ERROR] Statements directory not found: {statements_dir}")
        return []
        
    if not answers_dir.exists():
        print(f"[ERROR] Answers directory not found: {answers_dir}")
        return []
    
    print(f"[DATA] Loading statements from: {statements_dir}")
    
    samples = []
    
    # Look for statement files (pattern: statement_NNNN.txt)
    stmt_files = list(statements_dir.glob("statement_*.txt"))
    if not stmt_files:
        print("[WARNING] No statement_*.txt files found, trying all .txt files")
        stmt_files = list(statements_dir.glob("*.txt"))
    
    # Sort numerically by extracting number from filename
    def extract_number(filename):
        import re
        match = re.search(r'statement_(\d+)\.txt', filename.name)
        return int(match.group(1)) if match else 999999
    
    stmt_files = sorted(stmt_files, key=extract_number)[:n]
    
    print(f"[DATA] Found {len(stmt_files)} statement files, loading first {min(n, len(stmt_files))}")
    
    for stmt_file in stmt_files:
        try:
            # Read statement text
            with open(stmt_file, "r", encoding="utf-8") as f:
                statement = f.read().strip()
            
            # Find corresponding answer file
            answer_file = answers_dir / f"{stmt_file.stem}.json"
            
            if answer_file.exists():
                with open(answer_file, "r", encoding="utf-8") as f:
                    answer = json.load(f)
                samples.append((statement, answer))
            else:
                print(f"[WARNING] No answer file for {stmt_file.name}, creating dummy")
                # If no answer file, create dummy answer for testing
                dummy_answer = {"statement_is_true": True, "statement_topic": 1}
                samples.append((statement, dummy_answer))
                
        except Exception as e:
            print(f"[WARNING] Could not load {stmt_file}: {e}")
            continue
    
    print(f"[DATA] Successfully loaded {len(samples)} statement/answer pairs")
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Test RAG pipeline with custom embedding models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with PubMedBERT fine-tuned model
  python test_custom_embeddings.py --model models/pubmedbert-medical-final_20250806_165702
  
  # Test with BioBERT fine-tuned model
  python test_custom_embeddings.py --model models/biobert-medical-final_20250806_161455
  
  # Test with HuggingFace model
  python test_custom_embeddings.py --model pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
  
  # Test with custom settings
  python test_custom_embeddings.py --model models/pubmedbert-medical-final_20250806_165702 --n 10 --device cuda --verbose
        """
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to custom embedding model (local path or HuggingFace model ID)"
    )
    parser.add_argument(
        "--n", 
        type=int, 
        default=15, 
        help="Number of statements to test (default: 15)"
    )
    parser.add_argument(
        "--llm", 
        type=str, 
        default="cogito:8b", 
        help="LLM model to use (default: cogito:8b)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use: auto (default), cpu, or cuda"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--chunks-file",
        type=str,
        default="chunking/kg/chunks.jsonl",
        help="Path to chunks file (default: chunking/kg/chunks.jsonl)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Show detailed output for each statement"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("CUSTOM EMBEDDING MODEL RAG PIPELINE TEST")
    print("=" * 60)
    print(f"Model:           {args.model}")
    print(f"Statements:      {args.n}")
    print(f"LLM:             {args.llm}")
    print(f"Device:          {args.device}")
    print(f"Top-K:           {args.top_k}")
    print(f"Chunks file:     {args.chunks_file}")
    print()
    
    # Check if model path exists (for local models)
    model_path = args.model
    if not model_path.startswith("models/"):
        # Assume it's a relative path, make it absolute
        if not model_path.startswith("/") and not ":" in model_path:
            model_path = os.path.join(os.path.dirname(__file__), model_path)
    
    if os.path.exists(model_path):
        print(f"Local model found at: {model_path}")
    else:
        print(f"Assuming HuggingFace model: {args.model}")
        model_path = args.model
    
    # Initialize pipeline
    print("\nInitializing custom embedding pipeline...")
    start_init = time.time()
    
    try:
        pipeline = CustomEmbeddingRAGPipeline(
            embedding_model_path=model_path,
            llm_model=args.llm,
            device=args.device,
            top_k=args.top_k
        )
        
        # Load documents
        chunks_file = args.chunks_file
        if not os.path.isabs(chunks_file):
            chunks_file = os.path.join(os.path.dirname(__file__), chunks_file)
        
        pipeline.load_documents(chunks_file)
        
        init_time = time.time() - start_init
        print(f"Pipeline ready in {init_time:.1f}s\n")
        
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return 1
    
    # Load test statements
    print(f"Loading {args.n} test statements...")
    samples = load_training_statements(args.n)
    
    if not samples:
        print("No test statements found!")
        return 1
    
    print(f"Loaded {len(samples)} statements\n")
    
    # Test statements
    print(f"Testing {len(samples)} statements...\n")
    
    correct_binary = 0
    correct_topic = 0
    correct_both = 0
    total_time = 0
    
    for i, (statement, true_answer) in enumerate(samples):
        start_pred = time.time()
        
        try:
            # Make prediction
            pred_binary, pred_topic = pipeline.predict(statement)
            pred_time = time.time() - start_pred
            total_time += pred_time
            
            # Check correctness
            is_binary_correct = pred_binary == true_answer["statement_is_true"]
            is_topic_correct = pred_topic == true_answer["statement_topic"]
            
            if is_binary_correct:
                correct_binary += 1
            if is_topic_correct:
                correct_topic += 1
            if is_binary_correct and is_topic_correct:
                correct_both += 1
            
            # Display results
            if args.verbose:
                print(f"Statement {i + 1}:")
                print(f"  Text: '{statement[:100]}...'")
                print(f"  Binary: {pred_binary} (expected: {true_answer['statement_is_true']}) {'OK' if is_binary_correct else 'FAIL'}")
                print(f"  Topic:  {pred_topic} (expected: {true_answer['statement_topic']}) {'OK' if is_topic_correct else 'FAIL'}")
                print(f"  Time:   {pred_time:.2f}s")
                print()
            else:
                status = "OK-OK" if is_binary_correct and is_topic_correct else "OK-FAIL" if is_binary_correct else "FAIL-OK" if is_topic_correct else "FAIL-FAIL"
                print(f"[{i + 1:3d}/{len(samples)}] {status} Binary: {pred_binary}/{true_answer['statement_is_true']}, Topic: {pred_topic:3d}/{true_answer['statement_topic']:3d} ({pred_time:.1f}s)")
        
        except Exception as e:
            print(f"[{i + 1:3d}/{len(samples)}] ERROR: {str(e)[:60]}...")
    
    # Summary
    n = len(samples)
    avg_time = total_time / n if n > 0 else 0
    
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Embedding Model:    {args.model}")
    print(f"Binary Accuracy:    {correct_binary}/{n} ({correct_binary / n * 100:.1f}%)")
    print(f"Topic Accuracy:     {correct_topic}/{n} ({correct_topic / n * 100:.1f}%)")
    print(f"Combined Accuracy:  {correct_both}/{n} ({correct_both / n * 100:.1f}%)")
    print(f"Average Time:       {avg_time:.2f}s per statement")
    print(f"Total Time:         {total_time:.1f}s")
    print(f"{'=' * 60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())