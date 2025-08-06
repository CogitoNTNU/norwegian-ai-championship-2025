"""Batch processing pipeline for CUDA acceleration."""

import torch
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

from rag_pipeline_embeddings import EmbeddingsRAGPipeline


class BatchCUDARAGPipeline(EmbeddingsRAGPipeline):
    """RAG Pipeline optimized for batch processing with CUDA acceleration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cuda = kwargs.get('device') == 'cuda' and torch.cuda.is_available()

    def predict_batch(self, statements: List[str], batch_size: int = None) -> List[Tuple[int, int]]:
        """
        Process multiple statements in parallel batches for maximum CUDA efficiency.

        Args:
            statements: List of medical statements to classify
            batch_size: Batch size for parallel processing (auto if None)

        Returns:
            List of (statement_is_true, statement_topic) tuples
        """
        if batch_size is None:
            # Auto-select batch size based on device and memory
            if self.use_cuda:
                batch_size = min(32, len(statements))  # GPU optimal
            else:
                batch_size = min(8, len(statements))   # CPU safe

        print(f"🚀 Processing {len(statements)} statements in batches of {batch_size}")
        print(f"🔧 Device: {'CUDA' if self.use_cuda else 'CPU'}")

        results = []

        # Process in batches for optimal GPU utilization
        for i in tqdm(range(0, len(statements), batch_size), desc="Processing batches"):
            batch = statements[i:i + batch_size]

            if self.use_cuda and len(batch) > 1:
                # CUDA batch processing
                batch_results = self._cuda_batch_predict(batch)
            else:
                # Fallback to sequential processing
                batch_results = [self.predict(stmt) for stmt in batch]

            results.extend(batch_results)

            # Memory cleanup for CUDA
            if self.use_cuda and i % (batch_size * 4) == 0:
                torch.cuda.empty_cache()

        return results

    def _cuda_batch_predict(self, statements: List[str]) -> List[Tuple[int, int]]:
        """
        CUDA-optimized batch prediction with parallel embeddings.

        Args:
            statements: Batch of statements to process

        Returns:
            List of predictions for the batch
        """
        # Step 1: Batch embed ALL queries at once (major speedup)
        query_embeddings = self._batch_embed_queries(statements)

        # Step 2: Batch search for ALL queries simultaneously
        all_relevant_chunks = self._batch_search(query_embeddings, statements)

        # Step 3: Parallel LLM inference (if model supports it)
        results = []

        # Use thread pool for LLM calls (I/O bound with Ollama)
        with ThreadPoolExecutor(max_workers=min(4, len(statements))) as executor:
            # Submit all LLM tasks
            futures = []
            for i, statement in enumerate(statements):
                relevant_chunks = all_relevant_chunks[i]
                context = self._build_context(relevant_chunks)
                likely_topics = self._get_likely_topics(relevant_chunks)

                future = executor.submit(
                    self.llm_client.classify_statement,
                    statement, context, likely_topics
                )
                futures.append(future)

            # Collect results in order
            for future in futures:
                results.append(future.result())

        return results

    def _batch_embed_queries(self, queries: List[str]) -> np.ndarray:
        """
        Batch embed multiple queries at once for CUDA efficiency.

        Args:
            queries: List of query strings

        Returns:
            Batch of query embeddings
        """
        # This is where CUDA really shines - batch embedding generation
        if hasattr(self.document_store.embedding_model, "encode"):
            # Use model's native batch processing
            embeddings = self.document_store.embedding_model.encode(
                queries,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=len(queries) if self.use_cuda else 8
            )
        else:
            # Fallback to individual encoding
            embeddings = []
            for query in queries:
                emb = self.document_store.embedding_model.encode(
                    [query], convert_to_numpy=True, show_progress_bar=False
                )
                embeddings.append(emb[0])
            embeddings = np.array(embeddings)

        return embeddings

    def _batch_search(self, query_embeddings: np.ndarray, queries: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Batch search using vectorized FAISS operations.

        Args:
            query_embeddings: Batch of query embeddings
            queries: Original query strings (for fallback strategies)

        Returns:
            List of relevant chunks for each query
        """
        import faiss

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(query_embeddings)

        # Batch search all queries at once (FAISS vectorized operation)
        try:
            scores, indices = self.document_store.index.search(
                query_embeddings.astype("float32"),
                self.top_k
            )
        except Exception as e:
            print(f"[WARNING] Batch search failed: {e}, falling back to individual searches")
            # Fallback to individual searches
            all_results = []
            for i, query in enumerate(queries):
                results = self.document_store.search(query, self.top_k)
                all_results.append(results)
            return all_results

        # Convert batch results back to list format
        all_results = []
        for i in range(len(queries)):
            results = []
            for score, idx in zip(scores[i], indices[i]):
                if idx >= 0 and idx < len(self.document_store.chunks):
                    results.append({
                        "chunk": self.document_store.chunks[idx],
                        "metadata": self.document_store.chunk_metadata[idx],
                        "score": float(score),
                    })
            all_results.append(results)

        return all_results


class StreamingCUDARAGPipeline(BatchCUDARAGPipeline):
    """Streaming version for real-time applications with CUDA optimization."""

    def predict_streaming(self, statements: List[str], callback=None):
        """
        Stream predictions as they complete for real-time UX.

        Args:
            statements: List of statements to process
            callback: Function to call with each result (idx, result)
        """
        # Pre-warm CUDA embeddings with batch embedding
        print("🔥 Pre-warming CUDA embeddings...")
        _ = self._batch_embed_queries(statements[:min(4, len(statements))])

        # Process with overlapping batches for continuous GPU utilization
        batch_size = 8 if self.use_cuda else 4

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}

            # Submit overlapping batches
            for i in range(0, len(statements), batch_size):
                batch = statements[i:i + batch_size]
                future = executor.submit(self._cuda_batch_predict, batch)
                futures[future] = (i, batch)

            # Stream results as they complete
            for future in as_completed(futures):
                start_idx, batch = futures[future]
                try:
                    batch_results = future.result()
                    for j, result in enumerate(batch_results):
                        if callback:
                            callback(start_idx + j, result)
                        yield start_idx + j, result
                except Exception as e:
                    print(f"Batch failed: {e}")


# Performance testing utilities
def benchmark_cuda_vs_cpu(pipeline_cuda, pipeline_cpu, test_statements, runs=3):
    """Benchmark CUDA vs CPU performance."""
    import time

    cuda_times = []
    cpu_times = []

    print(f"🏁 Benchmarking {len(test_statements)} statements, {runs} runs each")

    for run in range(runs):
        # CUDA benchmark
        start = time.time()
        cuda_results = pipeline_cuda.predict_batch(test_statements)
        cuda_time = time.time() - start
        cuda_times.append(cuda_time)

        # CPU benchmark
        start = time.time()
        cpu_results = pipeline_cpu.predict_batch(test_statements)
        cpu_time = time.time() - start
        cpu_times.append(cpu_time)

        print(f"Run {run+1}: CUDA {cuda_time:.1f}s, CPU {cpu_time:.1f}s, Speedup: {cpu_time/cuda_time:.1f}x")

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_cuda = np.mean(cuda_times)
    avg_cpu = np.mean(cpu_times)
    speedup = avg_cpu / avg_cuda

    print(f"\n📊 BENCHMARK RESULTS:")
    print(f"   CUDA Average: {avg_cuda:.2f}s")
    print(f"   CPU Average:  {avg_cpu:.2f}s")
    print(f"   🚀 Speedup:   {speedup:.1f}x")

    return {
        'cuda_times': cuda_times,
        'cpu_times': cpu_times,
        'speedup': speedup,
        'cuda_avg': avg_cuda,
        'cpu_avg': avg_cpu
    }
