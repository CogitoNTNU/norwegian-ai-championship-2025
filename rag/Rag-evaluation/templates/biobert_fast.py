"""
Fast BioBERT Topic-classified BM25 RAG without NLI reranking.
Optimized for speed with <5s response times.
"""

import json
import os
import time
from typing import List, Dict, Any

import bm25s
import numpy as np
from sentence_transformers import SentenceTransformer


class BiobertFast:
    """Fast BioBERT Topic-classified BM25 RAG without NLI reranking."""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        
        # Load optimized indexes
        self._load_optimized_indexes()
        
        # Initialize BioBERT for topic classification only (no NLI reranking)
        print("🚀 Loading BioBERT for topic classification...")
        self.topic_classifier = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
        print("✅ BioBERT topic classifier loaded")

    def _load_optimized_indexes(self):
        """Load pre-computed optimized indexes."""
        base_dir = os.path.join(os.path.dirname(__file__), "..", "optimized_indexes")
        
        print(f"🚀 Loading optimized indexes from {base_dir}...")
        
        # Load document mapping from existing structure
        with open(os.path.join(base_dir, "document_mapping.json"), "r") as f:
            self.documents = json.load(f)
        
        # Load metadata
        with open(os.path.join(base_dir, "index_metadata.json"), "r") as f:
            metadata = json.load(f)
        
        # Build topic mapping and topic documents from document mapping
        self.topic_mapping = {}
        self.topic_documents = {}
        
        for doc in self.documents:
            topic_name = doc['topic_name']
            topic_id = doc['topic_id']
            
            # Build topic mapping
            self.topic_mapping[topic_name] = topic_id
            
            # Build topic documents mapping
            if topic_name not in self.topic_documents:
                self.topic_documents[topic_name] = []
            self.topic_documents[topic_name].append(doc['chunk_id'])
        
        # Load topic embeddings for fast topic classification
        topic_embeddings_path = os.path.join(base_dir, "topic_embeddings.npy")
        if os.path.exists(topic_embeddings_path):
            self.topic_embeddings = np.load(topic_embeddings_path)
            self.topic_names = list(self.topic_mapping.keys())
        else:
            # Fallback: compute topic embeddings on-the-fly
            self._compute_topic_embeddings()
        
        print(f"✅ Loaded optimized indexes:")
        print(f"   • {len(self.documents)} documents from {len(self.topic_mapping)} topics")
        print(f"   • {metadata['unique_articles']} unique articles")
        print(f"   • Average chunk size: {metadata['avg_chunk_words']:.1f} words")

    def _compute_topic_embeddings(self):
        """Compute embeddings for topic names for fast classification."""
        print("Computing topic embeddings...")
        self.topic_names = list(self.topic_mapping.keys())
        # Use topic names as simple text for embedding
        self.topic_embeddings = self.topic_classifier.encode(
            self.topic_names, 
            batch_size=32,
            show_progress_bar=False
        )

    def _classify_topic(self, query: str, top_k: int = 3) -> List[str]:
        """Fast topic classification using precomputed embeddings."""
        query_embedding = self.topic_classifier.encode([query], show_progress_bar=False)
        
        # Calculate cosine similarities
        similarities = np.dot(query_embedding, self.topic_embeddings.T).flatten()
        
        # Get top-k topics
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.topic_names[i] for i in top_indices]

    def _get_topic_documents(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Get documents for specified topics."""
        relevant_docs = []
        seen_doc_ids = set()
        
        for topic in topics:
            if topic in self.topic_documents:
                for chunk_id in self.topic_documents[topic]:
                    if chunk_id not in seen_doc_ids:
                        # Find the document with this chunk_id
                        for doc in self.documents:
                            if doc['chunk_id'] == chunk_id:
                                relevant_docs.append({
                                    'content': doc['text'],
                                    'article_title': doc['article_title'],
                                    'topic_name': doc['topic_name']
                                })
                                seen_doc_ids.add(chunk_id)
                                break
        
        return relevant_docs

    def _build_bm25_index(self, documents: List[Dict[str, Any]]) -> bm25s.BM25:
        """Build BM25 index for given documents."""
        if not documents:
            return None
            
        # Extract text content
        corpus = [doc['content'] for doc in documents]
        
        # Tokenize and build index
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        
        return retriever

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[str]:
        """Fast document retrieval using topic-classified BM25."""
        start_time = time.time()
        
        # Step 1: Fast topic classification (should be <0.5s)
        relevant_topics = self._classify_topic(query, top_k=3)
        topic_time = time.time() - start_time
        
        # Step 2: Get documents for relevant topics
        relevant_docs = self._get_topic_documents(relevant_topics)
        if not relevant_docs:
            return []
        
        docs_time = time.time() - start_time - topic_time
        
        # Step 3: Build BM25 index and retrieve (should be <2s)
        retriever = self._build_bm25_index(relevant_docs)
        if retriever is None:
            return []
        
        # Tokenize query and retrieve
        query_tokens = bm25s.tokenize([query], stopwords="en")
        results, scores = retriever.retrieve(query_tokens, k=min(top_k, len(relevant_docs)))
        
        # Extract top results
        contexts = []
        for i in range(len(results[0])):
            doc_idx = results[0][i]
            if doc_idx < len(relevant_docs):
                contexts.append(relevant_docs[doc_idx]['content'])
        
        retrieval_time = time.time() - start_time - topic_time - docs_time
        
        # Debug timing
        total_time = time.time() - start_time
        if total_time > 2.0:  # Only log if retrieval is slow
            print(f"Retrieval timing: topic={topic_time:.2f}s, docs={docs_time:.2f}s, bm25={retrieval_time:.2f}s, total={total_time:.2f}s")
        
        return contexts

    def run(self, user_input: str, reference_contexts: List[str] = None) -> Dict[str, Any]:
        """Run the fast BioBERT topic-classified BM25 RAG pipeline."""
        start_time = time.time()
        
        # Retrieve relevant documents
        retrieved_contexts = self.retrieve_documents(user_input, top_k=5)
        retrieval_time = time.time() - start_time
        
        if not retrieved_contexts:
            retrieved_contexts = ["No relevant medical information found."]
        
        # Prepare context for LLM
        context_str = "\n\n".join(retrieved_contexts)
        
        # Generate response using LLM
        llm_start = time.time()
        prompt = f"""You are a medical expert AI assistant. Based on the provided medical context, analyze the following statement and provide a JSON response.

Medical Context:
{context_str}

Statement to analyze: {user_input}

Please respond with a JSON object containing:
- "statement_is_true": 1 if the statement is medically accurate, 0 if false
- "statement_topic": a number from 0-114 representing the most relevant medical topic/specialty
- "explanation": brief explanation of your reasoning

Response:"""

        try:
            response = self.llm_client.query(prompt)
            llm_time = time.time() - llm_start
            total_time = time.time() - start_time
            
            # Log timing if over target
            if total_time > 5.0:
                print(f"⚠️ Slow response: retrieval={retrieval_time:.2f}s, llm={llm_time:.2f}s, total={total_time:.2f}s")
            
            return {
                "answer": response,
                "context": retrieved_contexts
            }
        except Exception as e:
            return {
                "answer": f'{{"statement_is_true": 1, "statement_topic": 0, "explanation": "Error: {str(e)}"}}',
                "context": retrieved_contexts
            }
