"""Advanced CUDA-optimized similarity search strategies."""

import faiss
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
import time


class CUDAEnhancedSearch:
    """Advanced GPU-accelerated similarity search with multiple optimization strategies."""

    def __init__(self, device="cuda"):
        self.device = device
        self.use_cuda = device == "cuda" and torch.cuda.is_available()
        self.gpu_resources = None

        if self.use_cuda:
            try:
                self.gpu_resources = faiss.StandardGpuResources()
                # Configure GPU memory pool for better performance
                self.gpu_resources.setDefaultNullStreamAllTempMemory(512 * 1024 * 1024)  # 512MB
                print("[CUDA] Advanced GPU resources initialized with optimized memory pool")
            except:
                print("[INFO] faiss-gpu not available, using CPU FAISS")
                self.use_cuda = False

    def create_optimized_index(self, embeddings: np.ndarray, index_type="auto") -> faiss.Index:
        """
        Create optimized FAISS index based on data size and GPU availability.

        Args:
            embeddings: Document embeddings to index
            index_type: "auto", "flat", "ivf", "hnsw", or "gpu_optimized"

        Returns:
            Optimized FAISS index
        """
        n_vectors, dimension = embeddings.shape

        print(f"🔧 Creating optimized index for {n_vectors} vectors, {dimension}D")

        if index_type == "auto":
            index_type = self._select_optimal_index_type(n_vectors, dimension)

        if self.use_cuda and index_type in ["flat", "ivf", "gpu_optimized"]:
            return self._create_gpu_index(embeddings, index_type, dimension)
        else:
            return self._create_cpu_index(embeddings, index_type, dimension)

    def _select_optimal_index_type(self, n_vectors: int, dimension: int) -> str:
        """Auto-select optimal index type based on dataset characteristics."""
        if n_vectors < 5000:
            return "flat"  # Exact search for small datasets
        elif n_vectors < 50000:
            return "ivf"   # IVF for medium datasets
        elif self.use_cuda:
            return "gpu_optimized"  # GPU-specific optimizations for large datasets
        else:
            return "hnsw"  # HNSW for large CPU datasets

    def _create_gpu_index(self, embeddings: np.ndarray, index_type: str, dimension: int) -> faiss.Index:
        """Create GPU-optimized FAISS index."""
        n_vectors = len(embeddings)

        if index_type == "flat":
            # GPU Flat index for exact search
            cpu_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine after normalization
            gpu_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)

        elif index_type == "ivf":
            # GPU IVF index for approximate search
            nlist = min(4096, max(64, int(np.sqrt(n_vectors))))  # Optimal cluster count
            quantizer = faiss.IndexFlatIP(dimension)
            cpu_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

            # Train on GPU
            gpu_quantizer = faiss.index_cpu_to_gpu(self.gpu_resources, 0, quantizer)
            gpu_index = faiss.IndexIVFFlat(gpu_quantizer, dimension, nlist)

            # Train with sample if dataset is large
            if n_vectors > 100000:
                train_size = min(100000, n_vectors)
                train_indices = np.random.choice(n_vectors, train_size, replace=False)
                train_embeddings = embeddings[train_indices]
            else:
                train_embeddings = embeddings

            print(f"[CUDA] Training IVF index with {len(train_embeddings)} samples...")
            faiss.normalize_L2(train_embeddings)
            gpu_index.train(train_embeddings.astype('float32'))

        elif index_type == "gpu_optimized":
            # Advanced GPU-specific index with multiple optimizations
            nlist = min(8192, max(128, int(np.sqrt(n_vectors))))

            # Use GPU-optimized quantizer with multiple GPUs if available
            quantizer = faiss.IndexFlatIP(dimension)
            cpu_index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 64, 8)  # PQ compression

            gpu_quantizer = faiss.index_cpu_to_gpu(self.gpu_resources, 0, quantizer)
            gpu_index = faiss.IndexIVFPQ(gpu_quantizer, dimension, nlist, 64, 8)

            # Train with large sample for better quality
            train_size = min(200000, n_vectors)
            if n_vectors > train_size:
                train_indices = np.random.choice(n_vectors, train_size, replace=False)
                train_embeddings = embeddings[train_indices]
            else:
                train_embeddings = embeddings

            print(f"[CUDA] Training GPU-optimized PQ index with {len(train_embeddings)} samples...")
            faiss.normalize_L2(train_embeddings)
            gpu_index.train(train_embeddings.astype('float32'))

        else:
            raise ValueError(f"Unsupported GPU index type: {index_type}")

        return gpu_index

    def _create_cpu_index(self, embeddings: np.ndarray, index_type: str, dimension: int) -> faiss.Index:
        """Create CPU-optimized FAISS index."""
        n_vectors = len(embeddings)

        if index_type == "flat":
            cpu_index = faiss.IndexFlatIP(dimension)

        elif index_type == "ivf":
            nlist = min(4096, max(64, int(np.sqrt(n_vectors))))
            quantizer = faiss.IndexFlatIP(dimension)
            cpu_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

            # Train
            faiss.normalize_L2(embeddings)
            cpu_index.train(embeddings.astype('float32'))

        elif index_type == "hnsw":
            # HNSW index for large CPU datasets
            cpu_index = faiss.IndexHNSWFlat(dimension, 32)  # M=32 connections
            cpu_index.hnsw.efConstruction = 200  # Higher quality construction

        else:
            raise ValueError(f"Unsupported CPU index type: {index_type}")

        return cpu_index

    def add_embeddings_optimized(self, index: faiss.Index, embeddings: np.ndarray, batch_size: int = None) -> None:
        """Add embeddings to index with optimal batching strategy."""
        if batch_size is None:
            if self.use_cuda:
                # GPU can handle larger batches
                batch_size = min(10000, len(embeddings))
            else:
                # CPU uses smaller batches
                batch_size = min(5000, len(embeddings))

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        print(f"🚀 Adding {len(embeddings)} embeddings in batches of {batch_size}...")

        for i in range(0, len(embeddings), batch_size):
            batch_end = min(i + batch_size, len(embeddings))
            batch_embeddings = embeddings[i:batch_end].astype('float32')

            if self.use_cuda:
                # For GPU, we can add larger batches efficiently
                index.add(batch_embeddings)
            else:
                # For CPU, smaller batches may be more memory efficient
                index.add(batch_embeddings)

            if i % (batch_size * 5) == 0:  # Progress every 5 batches
                print(f"   Added {batch_end}/{len(embeddings)} vectors...")

        print(f"✅ Index built with {index.ntotal} vectors")

    def search_optimized(self, index: faiss.Index, query_embeddings: np.ndarray, k: int = 10,
                        search_params: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimized search with dynamic parameters based on query characteristics.

        Args:
            index: FAISS index to search
            query_embeddings: Batch of query embeddings
            k: Number of results per query
            search_params: Optional search parameters

        Returns:
            (scores, indices) arrays
        """
        # Normalize queries
        faiss.normalize_L2(query_embeddings)

        # Set search parameters for approximate indices
        if search_params is None:
            search_params = self._get_optimal_search_params(index, len(query_embeddings), k)

        if hasattr(index, 'nprobe') and 'nprobe' in search_params:
            original_nprobe = index.nprobe
            index.nprobe = search_params['nprobe']

        if hasattr(index, 'hnsw') and 'ef' in search_params:
            original_ef = index.hnsw.efSearch
            index.hnsw.efSearch = search_params['ef']

        try:
            # Perform batch search
            start_time = time.time()
            scores, indices = index.search(query_embeddings.astype('float32'), k)
            search_time = time.time() - start_time

            queries_per_sec = len(query_embeddings) / search_time
            print(f"🔍 Searched {len(query_embeddings)} queries in {search_time:.3f}s ({queries_per_sec:.1f} QPS)")

        finally:
            # Restore original parameters
            if hasattr(index, 'nprobe') and 'nprobe' in search_params:
                index.nprobe = original_nprobe
            if hasattr(index, 'hnsw') and 'ef' in search_params:
                index.hnsw.efSearch = original_ef

        return scores, indices

    def _get_optimal_search_params(self, index: faiss.Index, n_queries: int, k: int) -> Dict:
        """Get optimal search parameters based on index type and query characteristics."""
        params = {}

        if hasattr(index, 'nprobe'):  # IVF index
            # Dynamic nprobe based on accuracy vs speed tradeoff
            if n_queries == 1:
                params['nprobe'] = 32  # Higher accuracy for single queries
            elif n_queries < 10:
                params['nprobe'] = 16  # Balanced for small batches
            else:
                params['nprobe'] = 8   # Speed optimized for large batches

        if hasattr(index, 'hnsw'):   # HNSW index
            # Dynamic ef based on k and batch size
            params['ef'] = max(k * 2, 64) if n_queries < 10 else max(k, 32)

        return params

    def benchmark_search_strategies(self, index: faiss.Index, test_queries: np.ndarray, k: int = 10) -> Dict:
        """Benchmark different search parameter configurations."""
        results = {}

        print("🏁 Benchmarking search strategies...")

        # Test different nprobe values for IVF indices
        if hasattr(index, 'nprobe'):
            for nprobe in [1, 4, 8, 16, 32]:
                params = {'nprobe': nprobe}
                start_time = time.time()
                scores, indices = self.search_optimized(index, test_queries, k, params)
                elapsed = time.time() - start_time

                results[f"IVF_nprobe_{nprobe}"] = {
                    'time': elapsed,
                    'qps': len(test_queries) / elapsed,
                    'scores': scores,
                    'indices': indices
                }

        # Test different ef values for HNSW indices
        if hasattr(index, 'hnsw'):
            for ef in [16, 32, 64, 128]:
                params = {'ef': ef}
                start_time = time.time()
                scores, indices = self.search_optimized(index, test_queries, k, params)
                elapsed = time.time() - start_time

                results[f"HNSW_ef_{ef}"] = {
                    'time': elapsed,
                    'qps': len(test_queries) / elapsed,
                    'scores': scores,
                    'indices': indices
                }

        # Print benchmark summary
        print("\n📊 SEARCH BENCHMARK RESULTS:")
        for config, metrics in results.items():
            print(f"   {config}: {metrics['qps']:.1f} QPS ({metrics['time']:.3f}s)")

        return results

    def cleanup(self):
        """Clean up GPU resources."""
        if self.gpu_resources is not None:
            self.gpu_resources = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("[CUDA] Search resources cleaned up")
