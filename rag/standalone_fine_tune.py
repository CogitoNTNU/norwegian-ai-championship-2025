"""Standalone fine-tuning script without any sentence-transformers dependencies."""

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

def load_medical_documents(topics_dir, topics_json):
    """Load medical documents and create chunks."""
    # Load topic mapping
    with open(topics_json, 'r') as f:
        topic_mapping = json.load(f)
    
    # Reverse mapping (name -> id)
    name_to_id = {name: topic_id for name, topic_id in topic_mapping.items()}
    
    chunks = []
    topics_path = Path(topics_dir)
    
    print("Loading medical documents...")
    for topic_dir in topics_path.iterdir():
        if topic_dir.is_dir() and topic_dir.name in name_to_id:
            topic_id = name_to_id[topic_dir.name]
            
            # Load all .md files in this topic
            for md_file in topic_dir.glob("*.md"):
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple chunking: split by paragraphs
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 100]
                
                for paragraph in paragraphs:
                    # Skip very short paragraphs and headers
                    if len(paragraph) > 200 and not paragraph.startswith('#'):
                        chunks.append({
                            'chunk': paragraph,
                            'metadata': {'topic_id': topic_id, 'topic_name': topic_dir.name}
                        })
    
    print(f"Loaded {len(chunks)} document chunks from {len(name_to_id)} topics")
    return chunks

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
        
        print(f"Organized chunks across {len(self.chunks_by_topic)} topics")
        
        # Create training examples
        self.examples = self._create_examples()
    
    def _create_examples(self):
        """Create positive and negative pairs."""
        examples = []
        
        for i, (statement, topic) in enumerate(zip(self.statements, self.topics)):
            # Get positive chunks (same topic)
            if topic in self.chunks_by_topic:
                positive_chunks = self.chunks_by_topic[topic]
                
                # Sample positive examples (1-2 per statement)
                num_positives = min(2, len(positive_chunks))
                for pos_chunk in random.sample(positive_chunks, num_positives):
                    examples.append({
                        'anchor': statement,
                        'positive': pos_chunk,
                        'label': 1.0  # Positive pair
                    })
                
                # Sample negative examples (2-3 per statement)
                negative_topics = [t for t in self.chunks_by_topic.keys() if t != topic]
                num_negatives = min(3, len(negative_topics))
                
                for neg_topic in random.sample(negative_topics, num_negatives):
                    neg_chunks = self.chunks_by_topic[neg_topic]
                    neg_chunk = random.choice(neg_chunks)
                    examples.append({
                        'anchor': statement,
                        'positive': neg_chunk,  # Actually negative, but using same structure
                        'label': 0.0  # Negative pair
                    })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize anchor and positive/negative
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

class MedicalEmbeddingModel(nn.Module):
    """Model for medical embedding learning."""
    
    def __init__(self, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        super().__init__()
        print(f"Loading base model: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Add projection head for better embeddings
        self.pooler = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 384),  # Final embedding dimension
            nn.LayerNorm(384)
        )
        
        print(f"Model initialized with {self.encoder.config.hidden_size} -> 384 embedding dimension")
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.pooler(cls_embedding)

def contrastive_loss(anchor_embeddings, positive_embeddings, labels, margin=0.2):
    """Contrastive loss function."""
    # Compute cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(anchor_embeddings, positive_embeddings)
    
    # Contrastive loss: pull positives together, push negatives apart
    pos_loss = (1 - labels) * torch.pow(torch.clamp(1 - cos_sim, min=0.0), 2)
    neg_loss = labels * torch.pow(torch.clamp(cos_sim - margin, min=0.0), 2)
    
    return torch.mean(pos_loss + neg_loss)

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
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
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0

def load_training_data(statements_dir, answers_dir, max_statements=200):
    """Load training statements and topics."""
    statements = []
    topics = []
    
    statements_path = Path(statements_dir)
    answers_path = Path(answers_dir)
    
    statement_files = sorted(list(statements_path.glob("statement_*.txt")))
    if max_statements:
        statement_files = statement_files[:max_statements]
    
    print(f"Loading {len(statement_files)} training statements...")
    
    for statement_file in statement_files:
        statement_id = statement_file.stem
        answer_file = answers_path / f"{statement_id}.json"
        
        if answer_file.exists():
            # Load statement
            with open(statement_file, 'r', encoding='utf-8') as f:
                statement = f.read().strip()
            
            # Load answer
            with open(answer_file, 'r') as f:
                answer = json.load(f)
            
            statements.append(statement)
            topics.append(answer['statement_topic'])
    
    print(f"Loaded {len(statements)} statements with topics")
    return statements, topics

def main():
    """Main fine-tuning pipeline."""
    print("=== Standalone Medical Embedding Fine-Tuning ===")
    
    # Configuration
    config = {
        'model_name': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        'statements_dir': 'data/processed/combined_train/statements',
        'answers_dir': 'data/processed/combined_train/answers',
        'topics_dir': 'data/raw/topics',
        'topics_json': 'data/topics.json',
        'max_statements': 150,  # Reasonable size for testing
        'batch_size': 4,  # Small for memory efficiency
        'epochs': 3,
        'learning_rate': 2e-5,
        'max_length': 256,  # Shorter for efficiency
        'warmup_steps': 50
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = MedicalEmbeddingModel(config['model_name']).to(device)
    
    # Load training data
    statements, topics = load_training_data(
        config['statements_dir'],
        config['answers_dir'],
        config['max_statements']
    )
    
    # Load document chunks
    document_chunks = load_medical_documents(config['topics_dir'], config['topics_json'])
    
    # Create dataset
    print("Creating training dataset...")
    dataset = MedicalContrastiveDataset(
        statements, topics, document_chunks, tokenizer, config['max_length']
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0  # No multiprocessing to avoid issues
    )
    
    print(f"Created dataset with {len(dataset)} training examples")
    print(f"Training batches: {len(dataloader)}")
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Train
    print("Starting training...")
    for epoch in range(config['epochs']):
        avg_loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{config['epochs']}, Average Loss: {avg_loss:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"models/medical_embeddings_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save complete model
    torch.save(model.state_dict(), output_dir / "model.pt")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    
    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Model saved to: {output_dir}")
    
    # Test the model
    print("Testing model...")
    model.eval()
    
    test_statements = [
        "Patient has chest pain and difficulty breathing",
        "Aspiration pneumonia affects elderly patients with dysphagia",
        "Cardiac arrest requires immediate CPR and defibrillation"
    ]
    
    for test_statement in test_statements:
        test_encoding = tokenizer(
            test_statement,
            truncation=True,
            padding='max_length',
            max_length=config['max_length'],
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            embedding = model(test_encoding['input_ids'], test_encoding['attention_mask'])
            print(f"Statement: '{test_statement[:50]}...'")
            print(f"Embedding shape: {embedding.shape}, Sample: {embedding[0][:3].cpu().numpy()}")
    
    print("Fine-tuning completed successfully!")
    return str(output_dir)

class StandaloneEmbeddingInference:
    """Inference class for the fine-tuned model."""
    
    def __init__(self, model_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        with open(Path(model_dir) / "config.json", 'r') as f:
            self.config = json.load(f)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(Path(model_dir) / "tokenizer")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = MedicalEmbeddingModel(self.config['model_name']).to(self.device)
        self.model.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location=self.device))
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
    try:
        model_path = main()
        
        print(f"\n{'='*50}")
        print("SUCCESS! Fine-tuning completed.")
        print(f"Model saved to: {model_path}")
        print(f"\nTo use your fine-tuned model:")
        print(f"from standalone_fine_tune import StandaloneEmbeddingInference")
        print(f"model = StandaloneEmbeddingInference('{model_path}')")
        print(f"embeddings = model.encode(['your medical text here'])")
        
    except Exception as e:
        print(f"Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()