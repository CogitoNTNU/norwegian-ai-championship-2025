import torch
import asyncio
import re
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import bm25s
import Stemmer
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import ollama

from . import config
from .data_loader import get_data_loader

logger = logging.getLogger(__name__)

class QueryGenerator:
    """Generate search queries from medical statements using Ollama."""
    
    def __init__(self, model_name: str = config.OLLAMA_MODEL):
        self.model_name = model_name
        
    async def generate_queries(self, statement: str) -> Dict[str, List[str]]:
        """Generate keywords and questions for retrieval."""
        prompt = f"""Medical statement: {statement}

Extract search terms and create questions. Return only valid JSON:

{{"keywords":["term1","term2","term3"],"questions":["question1?","question2?"]}}

JSON:"""
        
        try:
            response = ollama.generate(
                model=self.model_name, 
                prompt=prompt,
                options={"num_predict": config.QUERY_GENERATION_MAX_TOKENS}
            )
            content = response['response'].strip()
            
            # Find JSON content
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                json_str = content[start:end+1]
                result = json.loads(json_str)
                return {
                    'keywords': result.get('keywords', []),
                    'questions': result.get('questions', [])
                }
        except Exception as e:
            logger.warning(f"Query generation failed: {e}")
            
        # Fallback: extract keywords using regex
        words = re.findall(r'\b[a-zA-Z]{3,}\b', statement.lower())
        medical_words = [w for w in words if len(w) > 4][:5]
        return {
            'keywords': medical_words,
            'questions': [f"What is {statement}?"]
        }

class HybridRetriever:
    """Hybrid BM25s + FAISS retrieval system."""
    
    def __init__(self, k_sparse: int = config.K_SPARSE, k_dense: int = config.K_DENSE):
        self.k_sparse = k_sparse
        self.k_dense = k_dense
        
        # Initialize device for embeddings
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
            
        self.embedder = SentenceTransformer(
            config.EMBEDDER_MODEL, device=self.device
        )
        self.stemmer = Stemmer.Stemmer("english")
        
        # Load or build indices
        self.bm25, self.faiss_index, self.metadata = self._ensure_indices()
        
    def _ensure_indices(self) -> Tuple[Any, Any, pd.DataFrame]:
        """Load cached indices or build new ones."""
        
        # Check if cached indices exist
        if (config.BM25_CACHE.exists() and 
            config.FAISS_CACHE.exists() and 
            config.META_CACHE.exists()):
            logger.info("Loading cached indices...")
            return self._load_cached_indices()
        
        logger.info("Building indices from scratch...")
        return self._build_indices()
    
    def _load_cached_indices(self) -> Tuple[Any, Any, pd.DataFrame]:
        """Load indices from cache."""
        with open(config.BM25_CACHE, 'rb') as f:
            bm25 = pickle.load(f)
            
        faiss_index = faiss.read_index(str(config.FAISS_CACHE))
        
        with open(config.META_CACHE, 'rb') as f:
            metadata = pickle.load(f)
            
        return bm25, faiss_index, metadata
    
    def _build_indices(self) -> Tuple[Any, Any, pd.DataFrame]:
        """Build indices from documents."""
        
        # Load all statements
        texts = []
        data_loader = get_data_loader(config.DATASET_NAME)
        doc_chunks = data_loader.load_documents()
        texts = [doc['text'] for doc in doc_chunks]
        topic_ids = [doc['topic_id'] for doc in doc_chunks]
        
        logger.info(f"Processing {len(texts)} document chunks...")
        
        # Create metadata DataFrame with topic information
        metadata = pd.DataFrame({
            'id': range(len(texts)),
            'text': texts,
            'topic_id': topic_ids
        })
        
        # Build BM25s index
        logger.info("Building BM25s index...")
        tokenized = [[token for token in re.findall(r"\b\w+\b", text.lower())] 
                    for text in texts]
        stemmed = [self.stemmer.stemWords(tokens) for tokens in tokenized]
        
        bm25 = bm25s.BM25()
        bm25.index(stemmed)
        
        # Build FAISS index
        logger.info("Building FAISS index...")
        embeddings = self.embedder.encode(
            texts, batch_size=config.BATCH_SIZE, show_progress_bar=True
        )
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        faiss_index.add(embeddings.astype('float32'))
        
        # Cache the indices
        logger.info("Caching indices...")
        with open(config.BM25_CACHE, 'wb') as f:
            pickle.dump(bm25, f)
            
        faiss.write_index(faiss_index, str(config.FAISS_CACHE))
        
        with open(config.META_CACHE, 'wb') as f:
            pickle.dump(metadata, f)
            
        return bm25, faiss_index, metadata
    
    async def _bm25_search(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search using BM25s with stemming."""
        if not keywords:
            return []
            
        # Tokenize and stem keywords - BM25s expects List[List[str]] format
        query_tokens = [self.stemmer.stemWords([w.lower() for w in keywords])]
        
        # BM25s returns a named tuple (documents, scores) that can be unpacked
        documents, scores = self.bm25.retrieve(query_tokens, k=self.k_sparse)
        
        results = []
        # documents and scores are both 2D arrays: [query_index][result_index]
        for pos, idx in enumerate(documents[0]):  # We only have one query
            if idx < len(self.metadata):
                results.append({
                    'id': int(idx),
                    'text': self.metadata.iloc[idx]['text'],
                    'score': float(scores[0][pos]),  # scores[query_index][result_index]
                    'source': 'bm25'
                })
        return results
    
    async def _dense_search(self, query: str) -> List[Dict[str, Any]]:
        """Search using dense embeddings with FAISS."""
        if not query.strip():
            return []
            
        # Encode query
        query_embedding = self.embedder.encode([query])
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(
            query_embedding.astype('float32'), self.k_dense
        )
        
        results = []
        for pos, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append({
                    'id': int(idx),
                    'text': self.metadata.iloc[idx]['text'],
                    'score': float(scores[0][pos]),
                    'source': 'dense'
                })
        return results
    
    async def retrieve(self, queries: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining BM25s and dense search."""
        
        # Run searches in parallel
        sparse_task = self._bm25_search(queries.get('keywords', []))
        
        # Combine all questions into one query for dense search
        dense_query = " ".join(queries.get('questions', []))
        dense_task = self._dense_search(dense_query)
        
        sparse_results, dense_results = await asyncio.gather(sparse_task, dense_task)
        
        # Combine and deduplicate results
        seen_ids = set()
        combined_results = []
        
        # Add sparse results first (they tend to be more precise)
        for result in sparse_results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                combined_results.append(result)
        
        # Add dense results
        for result in dense_results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                combined_results.append(result)

        # Remove overly restrictive topic filtering that's causing poor retrieval
        # The classifier should determine topic from good context, not filtered context
        # if combined_results:
        #     # Use the topic from the top BM25 result as the filter topic
        #     top_bm25_topic = self.metadata.loc[combined_results[0]["id"], "topic_id"]
        #     
        #     # Filter results to only include documents with the same topic
        #     combined_results = [doc for doc in combined_results if self.metadata.loc[doc["id"], "topic_id"] == top_bm25_topic]

        return combined_results

class QwenReranker:
    """Qwen3-Reranker for cross-encoder reranking on Apple Silicon."""
    
    def __init__(self, model_name: str = config.RERANKER_MODEL):
        # Device selection with Apple Silicon optimization
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float16  # Faster on MPS
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            
        # Check if we should use lightweight reranker
        if config.USE_LIGHT_RERANKER:
            self.model = CrossEncoder("mixedbread-ai/mxbai-rerank-base-v1", device=self.device)
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, torch_dtype=self.dtype
            ).to(self.device).eval()
    
    async def rerank(self, query: str, documents: List[Dict[str, Any]], 
                    top_k: int = config.K_FINAL) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder."""
        
        if not documents:
            return []
            
        # Limit documents to avoid memory issues
        documents = documents[:100]  # Process max 100 docs
        
        if config.USE_LIGHT_RERANKER:
            # Use CrossEncoder (mixedbread) with limited candidates
            pairs = [(query, doc['text'][:config.MAX_CONTENT_LENGTH]) for doc in documents[:config.CANDIDATES_TO_RERANK]]
            scores = self.model.predict(pairs, batch_size=40)
            
            # Add scores to documents
            for doc, score in zip(documents[:config.CANDIDATES_TO_RERANK], scores):
                doc['rerank_score'] = float(score)
            
            # Give remaining documents a low score so they can still be sorted
            for doc in documents[config.CANDIDATES_TO_RERANK:]:
                doc['rerank_score'] = -1.0
        else:
            # Use Qwen3 reranker
            scores = []
            
            for doc in documents:
                text = doc['text'][:config.MAX_CONTENT_LENGTH]
                
                # Tokenize query-document pair
                inputs = self.tokenizer(
                    query, text, 
                    truncation=True, 
                    padding=True, 
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get reranking score
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Handle different output shapes - squeeze to get scalar
                    logits = outputs.logits.squeeze()
                    if logits.dim() == 0:  # Scalar case
                        score = torch.sigmoid(logits).item()
                    else:  # Vector case - take first element
                        score = torch.sigmoid(logits[0]).item()
                    scores.append(score)
            
            # Add scores to documents
            for doc, score in zip(documents, scores):
                doc['rerank_score'] = score
        
        # Sort by rerank score and return top-k
        ranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        return ranked_docs[:top_k]

class StatementClassifier:
    """Final classification using Ollama for statement truth and topic."""
    
    def __init__(self, model_name: str = config.OLLAMA_MODEL):
        self.model_name = model_name
        self.topic_mapping = config.load_topic_mapping()
        
        # Create reverse mapping for topic names to IDs
        self.topic_name_to_id = {name: idx for name, idx in self.topic_mapping.items()}
    
    async def classify(self, statement: str, context_docs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Classify statement as true/false and assign topic."""
        
        # Limit to top 3 chunks with shorter token limits
        top_chunks = context_docs[:config.CLASSIFICATION_MAX_CHUNKS]
        context_pieces = []
        for doc in top_chunks:
            # Truncate each chunk to ~120 tokens (roughly 480 characters)
            chunk_text = doc['text'][:480]
            context_pieces.append(chunk_text)
        
        context = "\n\n".join(context_pieces)
        
        # Focus primarily on statement content for topic classification
        # Context is secondary since retrieval may not always be perfect
        statement_lower = statement.lower()
        context_lower = context.lower() if context.strip() else ""
        
        # Weight statement much more heavily than context for topic selection
        primary_text = statement_lower
        secondary_text = context_lower
        
        # Define medical domain keywords and their associated topics
        domain_keywords = {
            "cardiac": [4, 7, 10, 11, 15, 22, 25, 38, 51, 49, 57, 77, 80, 82],  # Heart conditions
            "respiratory": [8, 13, 14, 21, 45, 46, 47, 59, 60, 61, 62, 63, 64, 65, 66, 67, 74, 81],  # Lung conditions
            "neuro": [18, 29, 35, 48, 71, 75, 79],  # Brain/nervous system
            "gi": [1, 2, 3, 6, 17, 37, 54, 56],  # Gastrointestinal
            "trauma": [0, 16, 20, 23, 26, 28, 39, 55, 68, 70, 79],  # Trauma/injury
            "infection": [9, 35, 36, 48, 72, 87, 89, 104],  # Infections/sepsis
            "emergency": [31, 32, 40, 42, 43, 44, 50, 52, 53, 58, 73, 76, 78],  # Emergency conditions
            "diagnostic": [83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]  # Tests/procedures
        }
        
        # Identify relevant topic domains based primarily on statement content
        relevant_topics = set()
        for keyword, topic_ids in domain_keywords.items():
            # Check statement first, then context
            if (keyword in primary_text or 
                any(word in primary_text for word in [
                    "heart", "cardiac", "chest", "mi", "stroke", "lung", "breath", "abdomen", 
                    "gi", "trauma", "injury", "infection", "sepsis", "emergency", "test", "lab"
                ])):
                relevant_topics.update(topic_ids)
            # Secondary check in context if nothing found in statement
            elif keyword in secondary_text:
                relevant_topics.update(topic_ids)
        
        # If no specific domain found, include most common emergency topics
        if not relevant_topics:
            relevant_topics = {1, 7, 16, 25, 27, 37, 61, 72, 75, 76, 83}
        
        # Create focused topic list
        topic_list = []
        sorted_topics = list(self.topic_mapping.items())
        for name, idx in sorted_topics:
            if idx in relevant_topics:
                topic_list.append(f"{idx}:{name.replace('_', '').replace(' ', '')}")
        
        # Limit to top 20 most relevant to keep prompt manageable
        topic_summary = " | ".join(topic_list[:20])
        
        # Enhanced prompt with better topic guidance and context emphasis
        prompt = f"""MEDICAL EMERGENCY CLASSIFICATION

Analyze the medical statement using the provided context. Return ONLY valid JSON.

TOPIC CATEGORIES:
{topic_summary}
...and 115 total topics (0-114)

RULES:
1. statement_is_true: 1 if statement is TRUE based on context, 0 if FALSE
2. statement_topic: Pick most relevant topic ID (0-114) based on medical content
3. Use context to determine accuracy - ignore if context is unrelated
4. MUST return complete JSON with both fields

EXAMPLES:

Statement: "Chest compressions depth should be 2+ inches"
Context: "AHA recommends 2-2.4 inch compression depth"
{{"statement_is_true": 1, "statement_topic": 18}}

Statement: "All MIs require immediate surgery"
Context: "Many MIs are treated with medications, not surgery"
{{"statement_is_true": 0, "statement_topic": 5}}

ANALYZE NOW:
Statement: {statement}
Context: {context}

JSON (required format):"""
        
        # Try to get classification with multiple parsing attempts
        result = await self._try_classification(prompt)
        if result:
            return result
        
        # Fallback classification - be conservative when unsure
        logger.warning("Using fallback classification")
        return {
            "statement_is_true": 0,  # Default to false when unsure (conservative)
            "statement_topic": 72    # Default to Sepsis_Septic Shock (common emergency)
        }
    
    async def _try_classification(self, prompt: str) -> Optional[Dict[str, int]]:
        """Try classification with robust JSON parsing."""
        try:
            response = ollama.generate(
                model=self.model_name, 
                prompt=prompt,
                options={"num_predict": config.CLASSIFICATION_MAX_TOKENS}
            )
            content = response['response'].strip()
            
            # Method 1: Standard JSON extraction
            result = self._extract_json_standard(content)
            if result:
                return result
            
            # Method 2: Regex-based extraction
            result = self._extract_json_regex(content)
            if result:
                return result
            
            # Method 3: Line-by-line parsing
            result = self._extract_json_lines(content)
            if result:
                return result
                
            logger.warning(f"All JSON extraction methods failed for content: {content[:200]}...")
            return None
            
        except Exception as e:
            logger.warning(f"Classification request failed: {e}")
            return None
    
    def _extract_json_standard(self, content: str) -> Optional[Dict[str, int]]:
        """Standard JSON extraction method."""
        try:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = content[start:end+1]
                result = json.loads(json_str)
                return self._validate_result(result)
        except Exception:
            pass
        return None
    
    def _extract_json_regex(self, content: str) -> Optional[Dict[str, int]]:
        """Regex-based JSON extraction."""
        try:
            # Look for JSON-like patterns - complete first
            patterns = [
                r'\{\s*"statement_is_true"\s*:\s*(\d+)\s*,\s*"statement_topic"\s*:\s*(\d+)\s*\}',
                r'\{\s*"statement_is_true"\s*:\s*(\d+)\s*,\s*"statement_topic"\s*:\s*(\d+)',
                r'statement_is_true["\s]*:\s*(\d+).*?statement_topic["\s]*:\s*(\d+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                if match:
                    try:
                        is_true = int(match.group(1))
                        topic = int(match.group(2))
                        result = {"statement_is_true": is_true, "statement_topic": topic}
                        return self._validate_result(result)
                    except (ValueError, IndexError):
                        continue
            
            # Fallback: Look for incomplete JSON and complete it
            incomplete_patterns = [
                r'\{\s*"statement_is_true"\s*:\s*(\d+)\s*\}',  # Missing topic
                r'"statement_is_true"\s*:\s*(\d+)',  # Just the value
            ]
            
            for pattern in incomplete_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    try:
                        is_true = int(match.group(1))
                        # Infer topic from context or use safe default
                        topic = self._infer_topic_from_context(content)
                        result = {"statement_is_true": is_true, "statement_topic": topic}
                        return self._validate_result(result)
                    except (ValueError, IndexError):
                        continue
                        
        except Exception:
            pass
        return None
    
    def _infer_topic_from_context(self, content: str) -> int:
        """Infer topic from content keywords when JSON is incomplete."""
        content_lower = content.lower()
        
        # Common medical topic keywords
        topic_keywords = {
            "heart": 5, "cardiac": 5, "mi": 5, "myocardial": 5,
            "stroke": 75, "brain": 75, "cerebral": 75,
            "chest": 31, "pain": 31, "angina": 31,
            "abdomen": 3, "abdominal": 3, "gi": 3,
            "trauma": 108, "injury": 108, "fracture": 28,
            "sepsis": 72, "infection": 72, "shock": 72,
            "lung": 45, "respiratory": 45, "ards": 45,
        }
        
        for keyword, topic_id in topic_keywords.items():
            if keyword in content_lower:
                return topic_id
        
        # Default to common emergency topic
        return 72  # Sepsis_Septic_Shock
    
    def _extract_json_lines(self, content: str) -> Optional[Dict[str, int]]:
        """Extract values from individual lines."""
        try:
            lines = content.split('\n')
            is_true = None
            topic = None
            
            for line in lines:
                # Look for true/false values
                if 'statement_is_true' in line.lower():
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        is_true = int(numbers[0])
                
                # Look for topic values
                if 'statement_topic' in line.lower():
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        topic = int(numbers[0])
            
            if is_true is not None and topic is not None:
                result = {"statement_is_true": is_true, "statement_topic": topic}
                return self._validate_result(result)
                
        except Exception:
            pass
        return None
    
    def _validate_result(self, result: Dict) -> Optional[Dict[str, int]]:
        """Validate the extracted result."""
        try:
            if (isinstance(result, dict) and 
                'statement_is_true' in result and 
                'statement_topic' in result and
                isinstance(result['statement_is_true'], int) and
                isinstance(result['statement_topic'], int) and
                0 <= result['statement_is_true'] <= 1 and
                0 <= result['statement_topic'] <= 114):
                return result
        except Exception:
            pass
        return None
