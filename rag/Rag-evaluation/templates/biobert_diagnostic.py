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

class BiobertDiagnostic:
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
            print(f"🚀 Loading optimized indexes from {optimized_dir}...")
            
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
                print(f"✅ Loaded optimized indexes:")
                print(f"   • {metadata['total_documents']:,} documents from {metadata['unique_topics']} topics")
                print(f"   • {metadata['unique_articles']} unique articles")
                print(f"   • Average chunk size: {metadata['avg_chunk_words']:.1f} words")
            else:
                print(f"✅ Loaded {len(self.documents)} documents from optimized indexes.")
        else:
            # Fallback: Load and compute indexes from scratch
            print("⚠️ Optimized indexes not found. Computing from scratch...")
            chunks_file_path = os.path.join(os.path.dirname(__file__), "..", "..", "chunking", "kg", "chunks.jsonl")
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

    def semantic_only_retrieval(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """BioBERT semantic search only"""
        if self.index is None:
            return []

        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        return [{'document': self.documents[idx], 'score': distances[0][i], 'rank': i+1} 
                for i, idx in enumerate(indices[0])]

    def bm25_only_retrieval(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """BM25s keyword search only"""
        if self.bm25_index is None:
            return []

        query_tokens = self._tokenize_and_stem(query)
        bm25_results_data = self.bm25_index.retrieve([query_tokens], k=k)
        bm25_results_indices = bm25_results_data.documents[0]
        bm25_scores = bm25_results_data.scores[0]
        return [{'document': self.documents[idx], 'score': score, 'rank': i+1} 
                for i, (idx, score) in enumerate(zip(bm25_results_indices, bm25_scores))]

    def _generate_contradictory_hypothesis(self, query: str) -> str:
        """Uses an LLM to generate a contradictory statement."""
        # This is a simplified example. In a real system, you might use a more sophisticated prompt.
        prompt = f"Please generate a single, concise sentence that directly contradicts the following statement: \"{query}\""
        
        # For this example, we'll simulate the LLM call with a simple rule-based transformation
        # to avoid the overhead of a real LLM call in this interactive setting.
        if " with " in query or " evidence of " in query:
            return query.replace(" with ", " without ").replace(" evidence of ", " an absence of ")
        return f"It is not the case that {query}"

    def _rerank_with_nli(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reranks results using a Natural Language Inference (NLI) model to find contradictions."""
        sentence_pairs = []
        for result in results:
            # Simple sentence splitting
            sentences = result['document']['text'].split('.')
            for sentence in sentences:
                if len(sentence) > 10: # Filter out very short non-sentences
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


    def topic_classified_bm25_retrieval(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Helper function to run the retrieval for a single query."""
        
        # Step 1: Use semantic search to classify the topic
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), 20)  # Get top 20 for topic classification and reranking
        
        # Get the most common topic from top semantic results
        topic_votes = {}
        for idx in indices[0]:
            topic = self.documents[idx].get('topic_name', 'Unknown')
            topic_votes[topic] = topic_votes.get(topic, 0) + 1
        
        # Select the most voted topic
        classified_topic = max(topic_votes, key=topic_votes.get)
        
        # Step 2: Filter documents to only include the classified topic
        topic_docs = []
        topic_doc_indices = []
        for i, doc in enumerate(self.documents):
            if doc.get('topic_name') == classified_topic:
                topic_docs.append(doc)
                topic_doc_indices.append(i)
        
        if not topic_docs:
            # Fallback to regular BM25 if no topic-specific docs found
            return self.bm25_only_retrieval(query, k)
        
        # Step 3: Create BM25 index for topic-specific documents
        topic_texts = [doc['text'] for doc in topic_docs]
        topic_tokens = [self._tokenize_and_stem(text) for text in topic_texts]
        
        topic_bm25_index = bm25s.BM25()
        topic_bm25_index.index(topic_tokens)
        
        # Step 4: Perform BM25 retrieval within the topic
        query_tokens = self._tokenize_and_stem(query)
        try:
            bm25_results_data = topic_bm25_index.retrieve([query_tokens], k=k*2) # Retrieve more for reranking
            bm25_results_indices = bm25_results_data.documents[0]
            bm25_scores = bm25_results_data.scores[0]
            
            results = []
            for i, (topic_idx, score) in enumerate(zip(bm25_results_indices, bm25_scores)):
                result = {
                    'document': topic_docs[topic_idx], 
                    'score': score, 
                    'rank': i+1,
                    'method': 'topic_classified_bm25',
                    'classified_topic': classified_topic,
                    'topic_doc_count': len(topic_docs)
                }
                results.append(result)

            # Step 5: Rerank results to prioritize contradictions using NLI
            reranked_results = self._rerank_with_nli(query, results)
            
            # Boost the score of the top NLI result
            if reranked_results:
                reranked_results[0]['score'] *= 2.0 # Double the score of the top NLI result

            return sorted(reranked_results, key=lambda x: x['score'], reverse=True)[:k]

        except Exception as e:
            print(f"Error in topic-classified BM25: {e}")
            return self.bm25_only_retrieval(query, k)

    def hybrid_retrieval(self, query: str, k: int = 5, k_rrf: int = 60) -> List[Dict[str, Any]]:
        """Hybrid retrieval with detailed diagnostics"""
        if self.index is None or self.bm25_index is None:
            return []

        # Semantic search
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k * 2)
        semantic_results = [{'document': self.documents[idx], 'score': distances[0][i], 'method': 'semantic', 'rank': i+1} 
                           for i, idx in enumerate(indices[0])]

        # Keyword search (BM25s)
        query_tokens = self._tokenize_and_stem(query)
        bm25_results_data = self.bm25_index.retrieve([query_tokens], k=k*2)
        bm25_results_indices = bm25_results_data.documents[0]
        bm25_scores = bm25_results_data.scores[0]
        bm25_results = [{'document': self.documents[idx], 'score': score, 'method': 'bm25', 'rank': i+1} 
                       for i, (idx, score) in enumerate(zip(bm25_results_indices, bm25_scores))]

        # RRF (Reciprocal Rank Fusion) with tracking
        fused_scores = {}
        
        # Process semantic results
        for rank, result in enumerate(semantic_results):
            chunk_id = result['document']['chunk_id']
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = {
                    'score': 0, 
                    'doc': result, 
                    'semantic_rank': rank + 1, 
                    'bm25_rank': None,
                    'semantic_contribution': 0,
                    'bm25_contribution': 0
                }
            contribution = 1 / (k_rrf + rank + 1)
            fused_scores[chunk_id]['score'] += contribution
            fused_scores[chunk_id]['semantic_contribution'] = contribution
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            chunk_id = result['document']['chunk_id']
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = {
                    'score': 0, 
                    'doc': result, 
                    'semantic_rank': None, 
                    'bm25_rank': rank + 1,
                    'semantic_contribution': 0,
                    'bm25_contribution': 0
                }
            else:
                fused_scores[chunk_id]['bm25_rank'] = rank + 1
            
            contribution = 1 / (k_rrf + rank + 1)
            fused_scores[chunk_id]['score'] += contribution
            fused_scores[chunk_id]['bm25_contribution'] = contribution
            
        sorted_results = sorted(fused_scores.values(), key=lambda x: x['score'], reverse=True)
        
        # Add final ranking and method information
        final_results = []
        for i, item in enumerate(sorted_results[:k]):
            result = item['doc'].copy()
            result['final_rank'] = i + 1
            result['rrf_score'] = item['score']
            result['semantic_rank'] = item['semantic_rank']
            result['bm25_rank'] = item['bm25_rank']
            result['semantic_contribution'] = item['semantic_contribution']
            result['bm25_contribution'] = item['bm25_contribution']
            
            # Determine primary method
            if item['semantic_contribution'] > item['bm25_contribution']:
                result['primary_method'] = 'semantic'
            elif item['bm25_contribution'] > item['semantic_contribution']:
                result['primary_method'] = 'bm25'
            else:
                result['primary_method'] = 'equal'
                
            final_results.append(result)
        
        return final_results

    def run(self, question: str, reference_contexts: List[str] = None) -> Dict[str, Any]:
        """Main method for the evaluation framework."""
        retrieved_docs = self.topic_classified_bm25_retrieval(question)
        
        context = ""
        for doc in retrieved_docs:
            context += f"Source [chunk_id: {doc['document']['chunk_id']}]: {doc['document']['text']}\n\n"

        # Get topic from retrieved documents
        topic = "Unknown"
        if retrieved_docs:
            topics = [doc['document'].get('topic_name', 'Unknown') for doc in retrieved_docs]
            topic = Counter(topics).most_common(1)[0][0]
        
        llm_response = self.llm_client.classify_statement(question, context)

        return {
            "answer": json.dumps({
                "statement_is_true": llm_response[0],
                "statement_topic": self.llm_client._topic_name_to_number(topic)
            }),
            "context": [doc['document']['text'] for doc in retrieved_docs]
        }

    def _extract_numbers(self, text: str) -> set:
        """Extracts all numbers (integers and floats) and percentages from a string."""
        # Regex to find numbers, including those with decimals and percentages
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        return set(numbers)

    def run(self, question: str, reference_contexts: List[str] = None) -> Dict[str, Any]:
        """Main evaluation method - simplified BM25 with NLI reranking"""
        k = 5  # Default number of documents to retrieve
        
        # Use topic-classified BM25 retrieval as the baseline
        retrieved_docs = self.topic_classified_bm25_retrieval(question, k)
        
        if not retrieved_docs:
            return {
                "answer": json.dumps({
                    "statement_is_true": False,
                    "statement_topic": 0
                }),
                "context": []
            }
        
        # Build context from retrieved documents
        context = ""
        for doc in retrieved_docs:
            context += f"Source [chunk_id: {doc['document']['chunk_id']}]: {doc['document']['text']}\n\n"

        # --- Numerical Contradiction Check ---
        question_numbers = self._extract_numbers(question)
        context_numbers = self._extract_numbers(context)

        # If the question contains numbers that are NOT in the context, it could be a contradiction.
        if question_numbers and not question_numbers.issubset(context_numbers):
            # This is a strong signal of a contradiction, so we can short-circuit the LLM call.
            return {
                "answer": json.dumps({
                    "statement_is_true": False, # Contradicted
                    "statement_topic": self._topic_name_to_number(self.classify_topic(retrieved_docs)),
                    "reason": f"Numerical mismatch detected. Question numbers {question_numbers} not found in context numbers {context_numbers}."
                }),
                "context": [doc['document']['text'] for doc in retrieved_docs]
            }
        # --- End Numerical Contradiction Check ---

        prompt = f"""
        You are a clinical fact-checker. Your task is to verify a medical statement against a provided context document and classify its topic.

        **CONTEXT DOCUMENT:**
        {context}

        **TASK:**
        Analyze the statement below. Follow these rules:
        1.  Find the single most relevant sentence in the context that supports or contradicts the statement.
        2.  **Pay strict attention to numerical values, percentages, and thresholds. The statement is contradicted if key values in the statement do not match the context.**
        3.  If the context is irrelevant or provides no evidence, state that in the 'explanation'.
        4.  Determine if the statement is contradicted.
        5.  Choose the most accurate topic ID from the topic list provided in the context.
        6.  Set 'statement_bool' to false if the context contradicts the statement OR if there is no relevant evidence.

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

    def run_all_retrievers(self, question: str, reference_contexts: List[str] = None) -> Dict[str, Any]:
        """Run diagnostic analysis comparing all retrieval methods"""
        
        # Get results from all four methods
        semantic_results = self.semantic_only_retrieval(question, k=5)
        bm25_results = self.bm25_only_retrieval(question, k=5)
        hybrid_results = self.hybrid_retrieval(question, k=5)
        topic_classified_bm25_results = self.topic_classified_bm25_retrieval(question, k=5)
        
        return {
            "question": question,
            "semantic_results": semantic_results,
            "bm25_results": bm25_results,
            "hybrid_results": hybrid_results,
            "topic_classified_bm25_results": topic_classified_bm25_results,
            "reference_contexts": reference_contexts
        }

    def analyze_question(self, question: str, ground_truth: str, reference_contexts: List[str] = None):
        """Detailed analysis of a single question"""
        diagnostic_data = self.run_all_retrievers(question, reference_contexts)
        
        print(f"\n{'='*80}")
        print(f"QUESTION: {question}")
        print(f"GROUND TRUTH: {ground_truth}")
        print(f"REFERENCE TOPIC: {reference_contexts[0] if reference_contexts else 'Unknown'}")
        print(f"{'='*80}")
        
        print(f"\n🔍 SEMANTIC SEARCH (BioBERT) TOP 5:")
        for i, result in enumerate(diagnostic_data['semantic_results']):
            doc = result['document']
            print(f"  {i+1}. Score: {result['score']:.4f}")
            print(f"     Topic: {doc.get('topic_name', 'Unknown')}")
            print(f"     Text: {doc['text'][:200]}...")
            print()
        
        print(f"\n🔑 BM25 SEARCH TOP 5:")
        for i, result in enumerate(diagnostic_data['bm25_results']):
            doc = result['document']
            print(f"  {i+1}. Score: {result['score']:.4f}")
            print(f"     Topic: {doc.get('topic_name', 'Unknown')}")
            print(f"     Text: {doc['text'][:200]}...")
            print()
        
        print(f"\n🔄 HYBRID SEARCH (RRF) TOP 5:")
        for i, result in enumerate(diagnostic_data['hybrid_results']):
            doc = result['document']
            print(f"  {i+1}. RRF Score: {result['rrf_score']:.4f}")
            print(f"     Primary Method: {result['primary_method']}")
            print(f"     Semantic Rank: {result['semantic_rank']} (contrib: {result['semantic_contribution']:.4f})")
            print(f"     BM25 Rank: {result['bm25_rank']} (contrib: {result['bm25_contribution']:.4f})")
            print(f"     Topic: {doc.get('topic_name', 'Unknown')}")
            print(f"     Text: {doc['text'][:200]}...")
            print()
        
        print(f"\n🧠 TOPIC-CLASSIFIED BM25 SEARCH TOP 5:")
        for i, result in enumerate(diagnostic_data['topic_classified_bm25_results']):
            doc = result['document']
            print(f"  {i+1}. Score: {result['score']:.4f}")
            print(f"     Classified Topic: {result.get('classified_topic', 'Unknown')} ({result.get('topic_doc_count', 0)} docs)")
            print(f"     Topic: {doc.get('topic_name', 'Unknown')}")
            print(f"     Text: {doc['text'][:200]}...")
            print()

        return diagnostic_data

    def classify_topic(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        if not retrieved_docs:
            return "Unknown"
        
        topic_names = [doc['document']['topic_name'] for doc in retrieved_docs]
        most_common_topic = Counter(topic_names).most_common(1)[0][0]
        return most_common_topic


    def _topic_name_to_number(self, topic_name: str) -> int:
        return self.llm_client._topic_name_to_number(topic_name)


class BiobertTopicBM25(BiobertDiagnostic):
    """
    A RAG template that specifically uses the topic-classified BM25 retrieval
    method with NLI reranking for evaluation.
    """
    def run(self, question: str, reference_contexts: List[str] = None) -> Dict[str, Any]:
        """
        Main evaluation method using only topic-classified BM25 with NLI reranking.
        """
        k = 5  # Default number of documents to retrieve
        
        # Step 1: Retrieve documents using the specific method
        retrieved_docs = self.topic_classified_bm25_retrieval(question, k)
        
        if not retrieved_docs:
            return {
                "answer": json.dumps({
                    "statement_is_true": 1, # Default to true
                    "statement_topic": 0
                }),
                "context": []
            }
        
        # Step 2: Build context from retrieved documents
        retrieved_contexts = [doc['document']['text'] for doc in retrieved_docs]
        combined_context = "\n\n".join(retrieved_contexts)

        # Step 3: Classify topic from retrieved docs
        topic = self.classify_topic(retrieved_docs)
        topic_num = self._topic_name_to_number(topic)

        # Step 4: Use LLM to classify the statement's truthfulness
        statement_is_true, _ = self.llm_client.classify_statement(question, combined_context)

        # Step 5: Format the final answer
        answer = {
            "statement_is_true": statement_is_true,
            "statement_topic": topic_num,
        }

        return {
            "answer": json.dumps(answer), 
            "context": retrieved_contexts
        }
