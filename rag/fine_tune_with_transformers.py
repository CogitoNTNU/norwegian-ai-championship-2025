"""Fine-tune embeddings using transformers directly (no sentence-transformers)."""

import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
import sys
import os

# Add rag-pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag-pipeline"))

from document_store_embeddings import EmbeddingsDocumentStore

class MedicalContrastiveDataset(Dataset):
    """Dataset for contrastive learning with medical statements."""
    
    def __init__(self, statements, topics, document_chunks, tokenizer, max_length=512):
        self.statements = statements
        self.topics = topics
        self.document_chunks = document_chunks
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Group chunks by topic for efficient sampling
        self.chunks_by_topic = {}
        for chunk in document_chunks:
            topic_id = chunk['metadata']['topic_id']
            if topic_id not in self.chunks_by_topic:
                self.chunks_by_topic[topic_id] = []
            self.chunks_by_topic[topic_id].append(chunk['chunk'])
        
        # Create training examples
        self.examples = self._create_examples()
    
    def _create_examples(self):
        """Create positive and negative pairs."""
        examples = []
        
        for i, (statement, topic) in enumerate(zip(self.statements, self.topics)):
            # Get positive chunks (same topic)
            if topic in self.chunks_by_topic:
                positive_chunks = self.chunks_by_topic[topic]
                
                # Sample positive examples
                for pos_chunk in random.sample(positive_chunks, min(2, len(positive_chunks))):
                    examples.append({
                        'anchor': statement,
                        'positive': pos_chunk,
                        'label': 1
                    })
                
                # Sample negative examples (different topics)
                negative_topics = [t for t in self.chunks_by_topic.keys() if t != topic]
                for neg_topic in random.sample(negative_topics, min(2, len(negative_topics))):
                    neg_chunks = self.chunks_by_topic[neg_topic]
                    neg_chunk = random.choice(neg_chunks)
                    examples.append({
                        'anchor': statement,
                        'positive': neg_chunk,
                        'label': 0
                    })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize anchor and positive
        anchor_encoding = self.tokenizer(
            example['anchor'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        positive_encoding = self.tokenizer(
            example['positive'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'anchor_input_ids': anchor_encoding['input_ids'].squeeze(),
            'anchor_attention_mask': anchor_encoding['attention_mask'].squeeze(),
            'positive_input_ids': positive_encoding['input_ids'].squeeze(),
            'positive_attention_mask': positive_encoding['attention_mask'].squeeze(),
            'label': torch.tensor(example['label'], dtype=torch.float)
        }

class ContrastiveEmbeddingModel(nn.Module):
    """Model for contrastive embedding learning."""
    
    def __init__(self, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooler = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 384),  # Final embedding dimension
            nn.LayerNorm(384)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.pooler(cls_embedding)

def contrastive_loss(anchor_embeddings, positive_embeddings, labels, margin=0.5):
    """Contrastive loss function."""
    # Compute cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(anchor_embeddings, positive_embeddings)
    
    # Contrastive loss
    pos_loss = (1 - labels) * torch.pow(1 - cos_sim, 2)
    neg_loss = labels * torch.pow(torch.clamp(cos_sim - margin, min=0.0), 2)
    
    return torch.mean(pos_loss + neg_loss)

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move to device
        anchor_ids = batch['anchor_input_ids'].to(device)
        anchor_mask = batch['anchor_attention_mask'].to(device)
        positive_ids = batch['positive_input_ids'].to(device)
        positive_mask = batch['positive_attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        anchor_embeddings = model(anchor_ids, anchor_mask)
        positive_embeddings = model(positive_ids, positive_mask)
        
        # Compute loss
        loss = contrastive_loss(anchor_embeddings, positive_embeddings, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def load_training_data(statements_dir, answers_dir, max_statements=200):
    """Load training statements and topics."""
    statements = []
    topics = []
    
    statements_path = Path(statements_dir)
    answers_path = Path(answers_dir)
    
    statement_files = sorted(list(statements_path.glob("statement_*.txt")))[:max_statements]
    
    for statement_file in statement_files:
        statement_id = statement_file.stem
        answer_file = answers_path / f"{statement_id}.json"
        
        if answer_file.exists():
            # Load statement
            with open(statement_file, 'r') as f:
                statement = f.read().strip()
            
            # Load answer
            with open(answer_file, 'r') as f:
                answer = json.load(f)
            
            statements.append(statement)
            topics.append(answer['statement_topic'])
    
    return statements, topics

def main():
    """Main fine-tuning pipeline."""
    print("=== Medical Embedding Fine-Tuning with Transformers ===")
    
    # Configuration
    config = {
        'model_name': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        'statements_dir': 'data/processed/combined_train/statements',
        'answers_dir': 'data/processed/combined_train/answers',
        'topics_dir': 'data/raw/topics',
        'topics_json': 'data/topics.json',
        'max_statements': 100,  # Small for testing
        'batch_size': 4,  # Small for memory
        'epochs': 2,
        'learning_rate': 2e-5,
        'max_length': 256
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading tokenizer and model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = ContrastiveEmbeddingModel(config['model_name']).to(device)
    
    # Load training data
    print("Loading training data...")
    statements, topics = load_training_data(
        config['statements_dir'],
        config['answers_dir'],
        config['max_statements']
    )
    
    # Load document chunks (reuse existing document store)
    print("Loading document chunks...")
    doc_store = EmbeddingsDocumentStore("pubmedbert-base-embeddings", device=device)
    doc_store.load_medical_documents(config['topics_dir'], config['topics_json'])
    
    # Get document chunks
    document_chunks = []
    for i, chunk_text in enumerate(doc_store.chunks):
        chunk_metadata = doc_store.chunk_metadata[i]
        document_chunks.append({
            'chunk': chunk_text,
            'metadata': chunk_metadata
        })
    
    print(f"Loaded {len(document_chunks)} document chunks")
    
    # Create dataset
    dataset = MedicalContrastiveDataset(
        statements, topics, document_chunks, tokenizer, config['max_length']
    )
    
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    
    print(f"Created dataset with {len(dataset)} examples")
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Train
    print("Starting training...")
    for epoch in range(config['epochs']):
        avg_loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{config['epochs']}, Average Loss: {avg_loss:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"models/transformers_embeddings_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    model.encoder.save_pretrained(output_dir / "encoder")
    torch.save(model.pooler.state_dict(), output_dir / "pooler.pt")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model saved to: {output_dir}")
    
    # Test the model
    print("Testing model...")
    model.eval()
    
    test_statement = "Patient has chest pain and difficulty breathing"
    test_encoding = tokenizer(
        test_statement,
        truncation=True,
        padding='max_length',
        max_length=config['max_length'],
        return_tensors='pt'
    ).to(device)
    
    with torch.no_grad():
        embedding = model(test_encoding['input_ids'], test_encoding['attention_mask'])
        print(f"Test embedding shape: {embedding.shape}")
        print(f"Embedding sample: {embedding[0][:5]}")
    
    print("Fine-tuning completed successfully!")
    return str(output_dir)

class TransformersEmbeddingInference:
    """Inference class for the fine-tuned transformers model."""
    
    def __init__(self, model_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(Path(model_dir) / "config.json", 'r') as f:
            self.config = json.load(f)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(Path(model_dir) / "tokenizer")
        
        # Load model
        self.model = ContrastiveEmbeddingModel(self.config['model_name']).to(self.device)
        
        # Load encoder weights
        encoder = AutoModel.from_pretrained(Path(model_dir) / "encoder")
        self.model.encoder = encoder.to(self.device)
        
        # Load pooler weights
        pooler_state = torch.load(Path(model_dir) / "pooler.pt", map_location=self.device)
        self.model.pooler.load_state_dict(pooler_state)
        
        self.model.eval()
        print(f"Loaded fine-tuned model from {model_dir}")
    
    def encode(self, texts, batch_size=8):
        """Encode texts to embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize batch
                encodings = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.config['max_length'],
                    return_tensors='pt'
                ).to(self.device)
                
                # Get embeddings
                batch_embeddings = self.model(
                    encodings['input_ids'],
                    encodings['attention_mask']
                )
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)

if __name__ == "__main__":
    model_path = main()
    
    print(f"\nTo use your fine-tuned model:")
    print(f"from fine_tune_with_transformers import TransformersEmbeddingInference")
    print(f"model = TransformersEmbeddingInference('{model_path}')")
    print(f"embeddings = model.encode(['your medical text here'])")