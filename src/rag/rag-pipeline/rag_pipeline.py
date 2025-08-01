from typing import Tuple
from document_store import DocumentStore
from llm_client import LocalLLMClient

class RAGPipeline:
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_model: str = "qwen3:8b",
                 top_k_retrieval: int = 3):
        """
        Initialize RAG pipeline.
        
        Args:
            embedding_model: Model for document embeddings
            llm_model: Local LLM model name
            top_k_retrieval: Number of relevant chunks to retrieve
        """
        self.document_store = DocumentStore(embedding_model)
        self.llm_client = LocalLLMClient(llm_model)
        self.top_k = top_k_retrieval
        
    def setup(self, topics_dir: str, topics_json: str, index_path: str = None) -> None:
        """
        Set up the RAG pipeline by loading documents and building index.
        
        Args:
            topics_dir: Path to medical topics directory
            topics_json: Path to topics mapping JSON
            index_path: Path to save/load pre-built index (optional)
        """
        print("Setting up RAG pipeline...")
        
        # Ensure LLM model is available
        self.llm_client.ensure_model_available()
        
        # Try to load existing index first
        if index_path:
            try:
                self.document_store.load_index(index_path)
                print("Loaded existing index")
                return
            except Exception as e:
                print(f"Could not load existing index: {e}")
                print("Building new index...")
        
        # Build new index
        self.document_store.load_medical_documents(topics_dir, topics_json)
        self.document_store.build_index()
        
        # Save index if path provided
        if index_path:
            self.document_store.save_index(index_path)
    
    def predict(self, statement: str) -> Tuple[int, int]:
        """
        Make prediction for a medical statement.
        
        Args:
            statement: Medical statement to classify
            
        Returns:
            Tuple of (statement_is_true, statement_topic)
        """
        # Step 1: Retrieve relevant context
        relevant_chunks = self.document_store.search(statement, k=self.top_k)
        
        # Step 2: Build context from retrieved chunks
        context = self._build_context(relevant_chunks)
        
        # Step 3: Use LLM to classify with context
        statement_is_true, statement_topic = self.llm_client.classify_statement(
            statement, context
        )
        
        return statement_is_true, statement_topic
    
    def _build_context(self, relevant_chunks: list) -> str:
        """Build context string from retrieved chunks."""
        if not relevant_chunks:
            return "No relevant medical context found."
        
        context_pieces = []
        seen_topics = set()
        
        for chunk_data in relevant_chunks:
            chunk = chunk_data['chunk']
            metadata = chunk_data['metadata']
            topic_name = metadata['topic_name']
            score = chunk_data['score']
            
            # Add topic diversity - don't repeat same topic too much
            if topic_name not in seen_topics or len(context_pieces) < 2:
                context_pieces.append(f"[{topic_name}] {chunk}")
                seen_topics.add(topic_name)
            
            # Limit total context length for speed
            if len('\n\n'.join(context_pieces)) > 1000:
                break
        
        return '\n\n'.join(context_pieces)
    
    def evaluate_on_training_data(self, training_statements_dir: str, 
                                 training_answers_dir: str, 
                                 max_samples: int = 10) -> dict:
        """
        Evaluate pipeline on training data for debugging.
        
        Args:
            training_statements_dir: Directory with statement files
            training_answers_dir: Directory with answer files
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        import os
        from pathlib import Path
        import json
        
        statements_path = Path(training_statements_dir)
        answers_path = Path(training_answers_dir)
        
        correct_binary = 0
        correct_topic = 0
        correct_both = 0
        total = 0
        
        statement_files = sorted(list(statements_path.glob("statement_*.txt")))[:max_samples]
        
        print(f"Evaluating on {len(statement_files)} samples...")
        
        for statement_file in statement_files:
            # Get corresponding answer file
            statement_id = statement_file.stem  # e.g., "statement_0001"
            answer_file = answers_path / f"{statement_id}.json"
            
            if not answer_file.exists():
                continue
            
            # Load statement and answer
            with open(statement_file, 'r') as f:
                statement = f.read().strip()
            
            with open(answer_file, 'r') as f:
                true_answer = json.load(f)
            
            # Make prediction
            try:
                pred_binary, pred_topic = self.predict(statement)
                
                # Check accuracy
                if pred_binary == true_answer['statement_is_true']:
                    correct_binary += 1
                
                if pred_topic == true_answer['statement_topic']:
                    correct_topic += 1
                
                if (pred_binary == true_answer['statement_is_true'] and 
                    pred_topic == true_answer['statement_topic']):
                    correct_both += 1
                
                total += 1
                
                print(f"{statement_id}: Binary {pred_binary}={true_answer['statement_is_true']} "
                      f"Topic {pred_topic}={true_answer['statement_topic']}")
                
            except Exception as e:
                print(f"Error processing {statement_file}: {e}")
        
        if total == 0:
            return {"error": "No samples processed"}
        
        return {
            "total_samples": total,
            "binary_accuracy": correct_binary / total,
            "topic_accuracy": correct_topic / total,
            "both_accuracy": correct_both / total,
            "correct_binary": correct_binary,
            "correct_topic": correct_topic,
            "correct_both": correct_both
        }
