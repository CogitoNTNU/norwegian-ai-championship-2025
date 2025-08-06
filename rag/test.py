import torch
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
from pathlib import Path

def load_medical_corpus():
    """Load the medical corpus with topic IDs and article content."""
    topics_file = "data/topics.json"
    topics_dir = "data/raw/topics"
    
    # Load topic mappings
    with open(topics_file, 'r') as f:
        topics = json.load(f)
    
    # Create reverse mapping (ID to topic name)
    id_to_topic = {v: k for k, v in topics.items()}
    
    # Load articles from each topic folder
    corpus = {}
    topic_mappings = {}
    
    for topic_name, topic_id in topics.items():
        topic_folder = Path(topics_dir) / topic_name
        if topic_folder.exists():
            for md_file in topic_folder.glob("*.md"):
                # Read article content
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract title and content
                lines = content.split('\n')
                title = None
                for line in lines:
                    if line.startswith('# ') and not line.startswith('## '):
                        title = line[2:].strip()
                        break
                
                if not title:
                    title = md_file.stem
                
                # Create corpus entry
                doc_id = f"{topic_id}_{md_file.stem}"
                corpus[doc_id] = {
                    'title': title,
                    'content': content[:2000],  # First 2000 chars
                    'topic_id': topic_id,
                    'topic_name': topic_name,
                    'file_path': str(md_file)
                }
                
                # Store topic mapping
                topic_mappings[doc_id] = topic_id
    
    return corpus, topics, id_to_topic, topic_mappings

def test_medical_retrieval():
    """Test the fine-tuned medical BioBERT embeddings on retrieval tasks."""
    
    print("🏥 Testing Medical RAG System with Fine-tuned BioBERT")
    print("=" * 70)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    # Load the fine-tuned model
    print("🔄 Loading fine-tuned model...")
    model = SentenceTransformer('biobert-medical-embeddings-mrl', truncate_dim=256)
    print("✅ Model loaded successfully!")
    print()
    
    # Load medical corpus
    print("📚 Loading medical corpus...")
    try:
        corpus, topics, id_to_topic, topic_mappings = load_medical_corpus()
        print(f"✅ Loaded {len(corpus)} medical articles from {len(topics)} topics")
    except Exception as e:
        print(f"❌ Error loading corpus: {e}")
        return
    print()
    
    # Create embeddings for corpus
    print("🧮 Creating embeddings for corpus...")
    doc_texts = [doc['content'] for doc in corpus.values()]
    doc_ids = list(corpus.keys())
    corpus_embeddings = model.encode(doc_texts, show_progress_bar=True)
    print("✅ Corpus embeddings created!")
    print()
    
    # Test queries with expected topics
    test_queries = [
        {
            "query": "Patient has chest pain radiating to left arm with ST elevation on ECG",
            "expected_topics": ["Acute Coronary Syndrome", "Acute Myocardial Infarction (STEMI_NSTEMI)"],
            "description": "Heart attack symptoms"
        },
        {
            "query": "Severe shortness of breath with bilateral lung infiltrates and oxygen saturation below 90%",
            "expected_topics": ["Acute Respiratory Distress Syndrome", "Heart Failure (Acute_Chronic)"],
            "description": "Respiratory distress"
        },
        {
            "query": "Diabetic patient with fruity breath odor, dehydration, and blood glucose over 400 mg/dL",
            "expected_topics": ["Diabetic Ketoacidosis"],
            "description": "Diabetic emergency"
        },
        {
            "query": "Sudden severe tearing chest pain radiating to back with blood pressure difference between arms",
            "expected_topics": ["Aortic Dissection"],
            "description": "Aortic emergency"
        },
        {
            "query": "Patient unconscious after allergic reaction with hypotension and urticaria",
            "expected_topics": ["Anaphylaxis"],
            "description": "Severe allergic reaction"
        }
    ]
    
    print("🎯 Testing Retrieval Performance")
    print("=" * 70)
    
    total_correct = 0
    total_queries = len(test_queries)
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected_topics = test_case["expected_topics"]
        description = test_case["description"]
        
        print(f"\n📝 Query {i}: {description}")
        print(f"Query: {query}")
        print(f"Expected topics: {', '.join(expected_topics)}")
        print("-" * 50)
        
        # Encode query
        query_embedding = model.encode([query])
        
        # Calculate similarities
        similarities = model.similarity(query_embedding, corpus_embeddings)[0]
        
        # Get top 5 results
        top_k = 5
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        print("🔍 Top retrieval results:")
        found_correct_topic = False
        
        for rank, idx in enumerate(top_indices, 1):
            doc_id = doc_ids[idx]
            doc = corpus[doc_id]
            similarity_score = similarities[idx]
            
            # Check if this matches expected topic
            is_correct = doc['topic_name'] in expected_topics
            if is_correct and rank <= 3:  # Consider top 3 as success
                found_correct_topic = True
            
            status = "✅" if is_correct else "❌"
            print(f"  {rank}. {status} [{similarity_score:.4f}] Topic: {doc['topic_name']}")
            print(f"     Title: {doc['title']}")
            print(f"     Article ID: {doc_id}")
            print()
        
        if found_correct_topic:
            total_correct += 1
            print("🎉 SUCCESS: Found correct topic in top 3!")
        else:
            print("😞 MISS: Expected topic not found in top 3")
        
        print("="*50)
    
    # Calculate overall accuracy
    accuracy = (total_correct / total_queries) * 100
    print(f"\n🎯 OVERALL RESULTS")
    print(f"Queries tested: {total_queries}")
    print(f"Correct retrievals: {total_correct}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    # Model performance summary
    print(f"\n📊 MODEL PERFORMANCE SUMMARY")
    print(f"Model: Fine-tuned BioBERT Medical Embeddings")
    print(f"Embedding dimension: 256")
    print(f"Corpus size: {len(corpus)} articles")
    print(f"Topics covered: {len(topics)}")
    print(f"Retrieval accuracy: {accuracy:.1f}%")
    
    if accuracy >= 80:
        print("🌟 EXCELLENT: Model shows strong medical domain understanding!")
    elif accuracy >= 60:
        print("👍 GOOD: Model shows decent medical retrieval performance")
    else:
        print("🔧 NEEDS IMPROVEMENT: Consider more fine-tuning")

if __name__ == "__main__":
    test_medical_retrieval()