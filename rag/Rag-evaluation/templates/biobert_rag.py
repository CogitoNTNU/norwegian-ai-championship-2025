
import os
import json
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from llm_client import LocalLLMClient
from collections import Counter

import re
import bm25s
import Stemmer

class BiobertRAG:
    def __init__(self, llm_client: LocalLLMClient = None):
        self.llm_client = llm_client or LocalLLMClient()
        self.embedding_model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
        self.nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
        self.documents = []
        self.document_texts = []
        self.embeddings = None
        self.index = None
        self.bm25_index = None
        self.stemmer = Stemmer.Stemmer("english")
        self._load_and_embed_data()

    def _load_and_embed_data(self):
        # Try to load optimized indexes first
        optimized_dir = os.path.join(os.path.dirname(__file__), "..", "optimized_indexes")
        faiss_index_path = os.path.join(optimized_dir, "biobert_faiss.index")
        bm25_index_path = os.path.join(optimized_dir, "bm25_index.pkl")
        mapping_path = os.path.join(optimized_dir, "document_mapping.json")
        metadata_path = os.path.join(optimized_dir, "index_metadata.json")
        
        if all(os.path.exists(p) for p in [faiss_index_path, bm25_index_path, mapping_path]):
            print(f"🚀 Loading optimized BioBERT indexes from {optimized_dir}...")
            
            # Load FAISS index
            self.index = faiss.read_index(faiss_index_path)
            
            # Load BM25 index
            import pickle
            with open(bm25_index_path, "rb") as f:
                self.bm25_index = pickle.load(f)
            
            # Load document mapping
            with open(mapping_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            
            self.document_texts = [doc['text'] for doc in self.documents]
            
            # Load and display metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                print(f"✅ Loaded optimized BioBERT indexes:")
                print(f"   • {metadata['total_documents']:,} documents from {metadata['unique_topics']} topics")
                print(f"   • {metadata['unique_articles']} unique articles")
                print(f"   • Average chunk size: {metadata['avg_chunk_words']:.1f} words")
            else:
                print(f"✅ Loaded {len(self.documents)} documents from optimized indexes.")
        else:
            # Fallback: Load and compute indexes from scratch
            print("⚠️ Optimized BioBERT indexes not found. Computing from scratch...")
            chunks_file_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "chunks.jsonl")
            if not os.path.exists(chunks_file_path):
                print(f"❌ chunks.jsonl file not found at {chunks_file_path}")
                return

            print(f"📄 Loading documents from {chunks_file_path}...")
            with open(chunks_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.documents.append(json.loads(line))
            
            if self.documents:
                self.document_texts = [doc['text'] for doc in self.documents]
                
                # Create FAISS index
                print("🧠 Computing BioBERT embeddings...")
                self.embeddings = self.embedding_model.encode(self.document_texts, show_progress_bar=True)
                self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
                self.index.add(self.embeddings.astype('float32'))
                
                # Create BM25 index
                print("🔤 Creating BM25 index...")
                tokenized_corpus = [self._tokenize_and_stem(doc) for doc in self.document_texts]
                self.bm25_index = bm25s.BM25()
                self.bm25_index.index(tokenized_corpus)
                
                print(f"✅ Created indexes for {len(self.documents)} documents.")

    def _tokenize_and_stem(self, text: str) -> List[str]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        return self.stemmer.stemWords(tokens)

    def _rerank_with_nli(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reranks results using Natural Language Inference to prioritize contradictions."""
        if not results:
            return results
            
        sentence_pairs = []
        for result in results:
            # Simple sentence splitting
            sentences = result['document']['text'].split('.')
            for sentence in sentences:
                if len(sentence) > 10:  # Filter out very short non-sentences
                    sentence_pairs.append([query, sentence.strip()])

        if not sentence_pairs:
            return results

        # The NLI model provides scores for [contradiction, entailment, neutral]
        scores = self.nli_model.predict(sentence_pairs)
        
        # Assign the highest contradiction score to each document
        doc_to_max_contradiction_score = {}
        pair_idx = 0
        for result in results:
            sentences = result['document']['text'].split('.')
            max_contradiction_score = -1
            for sentence in sentences:
                if len(sentence) > 10:
                    # The first score in the output is for contradiction
                    contradiction_score = scores[pair_idx][0]
                    if contradiction_score > max_contradiction_score:
                        max_contradiction_score = contradiction_score
                    pair_idx += 1
            doc_id = result['document']['chunk_id']
            doc_to_max_contradiction_score[doc_id] = max_contradiction_score

        # Update the score of each result with the NLI contradiction score
        for result in results:
            doc_id = result['document']['chunk_id']
            result['nli_contradiction_score'] = doc_to_max_contradiction_score.get(doc_id, -1)

        return sorted(results, key=lambda x: x['nli_contradiction_score'], reverse=True)

    def hybrid_retrieval(self, query: str, k: int = 5, k_rrf: int = 60) -> List[Dict[str, Any]]:
        """Hybrid retrieval with NLI reranking for contradiction detection."""
        if self.index is None or self.bm25_index is None:
            return []

        # Semantic search
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k * 2)
        semantic_results = [{'document': self.documents[idx], 'score': distances[0][i]} for i, idx in enumerate(indices[0])]

        # Keyword search (BM25s) - using the proper bm25s library
        query_tokens = self._tokenize_and_stem(query)
        bm25_results_data = self.bm25_index.retrieve([query_tokens], k=k*2)
        bm25_results_indices = bm25_results_data.documents[0]
        bm25_scores = bm25_results_data.scores[0]
        bm25_results = [{'document': self.documents[idx], 'score': score} for idx, score in zip(bm25_results_indices, bm25_scores)]

        # RRF (Reciprocal Rank Fusion)
        fused_scores = {}
        
        # Process semantic results
        for rank, result in enumerate(semantic_results):
            chunk_id = result['document']['chunk_id']
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = {'score': 0, 'doc': result}
            fused_scores[chunk_id]['score'] += 1 / (k_rrf + rank + 1)
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            chunk_id = result['document']['chunk_id']
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = {'score': 0, 'doc': result}
            fused_scores[chunk_id]['score'] += 1 / (k_rrf + rank + 1)
            
        sorted_results = sorted(fused_scores.values(), key=lambda x: x['score'], reverse=True)
        initial_results = [item['doc'] for item in sorted_results[:k*2]]  # Get more for reranking
        
        # Apply NLI reranking to prioritize contradictions
        nli_reranked = self._rerank_with_nli(query, initial_results)
        
        # Boost the score of the top NLI result
        if nli_reranked:
            nli_reranked[0]['score'] *= 2.0  # Double the score of the top NLI result
        
        return nli_reranked[:k]

    def classify_topic(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        if not retrieved_docs:
            return "Unknown"
        
        topic_names = [doc['document']['topic_name'] for doc in retrieved_docs]
        most_common_topic = Counter(topic_names).most_common(1)[0][0]
        return most_common_topic

    def run(self, question: str, reference_contexts: List[str] = None) -> Dict[str, Any]:
        retrieved_docs = self.hybrid_retrieval(question)
        
        context = ""
        for doc in retrieved_docs:
            context += f"Source [chunk_id: {doc['document']['chunk_id']}]: {doc['document']['text']}\n\n"

        prompt = f"""
        You are a clinical fact-checker. Your task is to verify a medical statement against a provided context document and classify its topic.

        **CONTEXT DOCUMENT:**
        {context}

        **TASK:**
        Analyze the statement below. Follow these rules:
        1.  Find the single most relevant sentence in the context that supports or contradicts the statement.
        2.  If the context is irrelevant or provides no evidence, state that in the 'explanation'.
        3.  Determine if the statement is contradicted.
        4.  Choose the most accurate topic ID from the topic list provided in the context.
        5.  Set 'statement_bool' to false if the context contradicts the statement OR if there is no relevant evidence.

        --- 
        **EXAMPLE 1 (Contradiction):**
        **Statement:** "All fevers are dangerous and require immediate medical attention."
        **Context:** "While high-grade fevers can be serious, most low-grade fevers are self-resolving..."
        **Answer:**
        ```json
        {{
          "explanation": "The context states that 'most low-grade fevers are self-resolving', which contradicts the idea that all fevers require immediate medical attention.",
          "is_contradicted": "yes",
          "statement_bool": false,
          "statement_topic": 44
        }}
        ```

        **EXAMPLE 2 (No Evidence):**
        **Statement:** "Neonatal testicular torsion is caused by a viral infection."
        **Context:** "Ambulatory ECG monitoring is an essential diagnostic tool... it can be critical for diagnosing arrhythmias."
        **Answer:**
        ```json
        {{
          "explanation": "The provided context about ECG monitoring does not contain information about testicular torsion.",
          "is_contradicted": "unverifiable",
          "statement_bool": false,
          "statement_topic": 78
        }}
        ```

        **FINAL TASK:**
        **Statement:** {question}
        **Answer:**
        """
        
        topic = self.classify_topic(retrieved_docs)
        llm_response = self.llm_client.classify_statement(question, context)

        return {
            "answer": json.dumps({
                "statement_is_true": llm_response[0],
                "statement_topic": self._topic_name_to_number(topic)
            }),
            "context": [doc['document']['text'] for doc in retrieved_docs]
        }

    def _topic_name_to_number(self, topic_name: str) -> int:
        return self.llm_client._topic_name_to_number(topic_name)


