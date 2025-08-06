"""CUDA-optimized caching and memory management for RAG pipeline."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import hashlib
import pickle
import time
from pathlib import Path
from dataclasses import dataclass
from collections import OrderedDict
import threading


@dataclass
class CacheStats:
    """Statistics for cache performance."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_mb: float = 0
    gpu_memory_mb: float = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0


class CUDAMemoryManager:
    """Intelligent CUDA memory management for RAG operations."""

    def __init__(self, device="cuda", max_gpu_memory_gb=6.0):
        """
        Initialize CUDA memory manager.

        Args:
            device: Target device
            max_gpu_memory_gb: Maximum GPU memory to use (GB)
        """
        self.device = device
        self.use_cuda = device == "cuda" and torch.cuda.is_available()
        self.max_gpu_memory = max_gpu_memory_gb * 1024 * 1024 * 1024  # Convert to bytes

        self.allocated_memory = 0
        self.memory_blocks = {}  # Track allocated blocks
        self.lock = threading.Lock()

        if self.use_cuda:
            self.gpu_properties = torch.cuda.get_device_properties(0)
            self.total_gpu_memory = self.gpu_properties.total_memory
            print(f"[CUDA] GPU Memory Manager initialized:")
            print(f"   Total GPU Memory: {self.total_gpu_memory / 1024**3:.1f}GB")
            print(f"   Max Usage Limit: {max_gpu_memory_gb:.1f}GB")

    def allocate_embedding_cache(self, cache_size_mb: int, embedding_dim: int) -> Optional[torch.Tensor]:
        """Allocate GPU memory for embedding cache."""
        if not self.use_cuda:
            return None

        cache_size_bytes = cache_size_mb * 1024 * 1024

        with self.lock:
            if self.allocated_memory + cache_size_bytes > self.max_gpu_memory:
                print(f"[CUDA] Not enough GPU memory for cache ({cache_size_mb}MB)")
                return None

            try:
                # Calculate number of embeddings that fit in cache
                embedding_bytes = embedding_dim * 4  # float32
                max_embeddings = cache_size_bytes // embedding_bytes

                cache_tensor = torch.zeros(
                    (max_embeddings, embedding_dim),
                    dtype=torch.float32,
                    device=self.device
                )

                self.allocated_memory += cache_size_bytes
                cache_id = f"embedding_cache_{len(self.memory_blocks)}"
                self.memory_blocks[cache_id] = {
                    'tensor': cache_tensor,
                    'size_bytes': cache_size_bytes,
                    'type': 'embedding_cache'
                }

                print(f"[CUDA] Allocated {cache_size_mb}MB embedding cache for {max_embeddings} embeddings")
                return cache_tensor

            except RuntimeError as e:
                print(f"[CUDA] Failed to allocate cache: {e}")
                return None

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        stats = {
            'allocated_mb': self.allocated_memory / (1024 * 1024),
            'max_allowed_mb': self.max_gpu_memory / (1024 * 1024),
            'num_blocks': len(self.memory_blocks)
        }

        if self.use_cuda:
            stats.update({
                'gpu_allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
                'gpu_reserved_mb': torch.cuda.memory_reserved() / (1024 * 1024),
                'gpu_total_mb': self.total_gpu_memory / (1024 * 1024)
            })

        return stats

    def cleanup(self):
        """Clean up all allocated memory."""
        with self.lock:
            for block_id, block_info in self.memory_blocks.items():
                del block_info['tensor']  # Release tensor

            self.memory_blocks.clear()
            self.allocated_memory = 0

            if self.use_cuda:
                torch.cuda.empty_cache()
                print("[CUDA] Memory manager cleaned up")


class CUDAEmbeddingCache:
    """High-performance embedding cache with CUDA optimization."""

    def __init__(self, cache_size_mb=512, embedding_dim=768, device="cuda", ttl_seconds=3600):
        """
        Initialize CUDA embedding cache.

        Args:
            cache_size_mb: Cache size in MB
            embedding_dim: Dimension of embeddings
            device: Target device
            ttl_seconds: Time-to-live for cache entries
        """
        self.device = device
        self.use_cuda = device == "cuda" and torch.cuda.is_available()
        self.embedding_dim = embedding_dim
        self.ttl_seconds = ttl_seconds

        # Memory manager
        self.memory_manager = CUDAMemoryManager(device)

        # CPU-based metadata cache (small and fast)
        self.metadata_cache = OrderedDict()  # {hash: (index, timestamp, query)}
        self.max_entries = (cache_size_mb * 1024 * 1024) // (embedding_dim * 4)  # float32

        # GPU embedding cache (if available)
        self.gpu_cache = None
        if self.use_cuda:
            self.gpu_cache = self.memory_manager.allocate_embedding_cache(
                cache_size_mb, embedding_dim
            )

        # Fallback CPU cache
        self.cpu_cache = np.zeros((self.max_entries, embedding_dim), dtype=np.float32)

        # Cache statistics
        self.stats = CacheStats()
        self.lock = threading.Lock()

        print(f"🗄️  Embedding cache initialized: {self.max_entries} slots, {cache_size_mb}MB")
        print(f"   GPU Cache: {'✅' if self.gpu_cache is not None else '❌'}")

    def _hash_query(self, query: str) -> str:
        """Generate hash for query string."""
        return hashlib.md5(query.encode()).hexdigest()

    def get(self, query: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        query_hash = self._hash_query(query)
        current_time = time.time()

        with self.lock:
            if query_hash in self.metadata_cache:
                index, timestamp, original_query = self.metadata_cache[query_hash]

                # Check TTL
                if current_time - timestamp > self.ttl_seconds:
                    # Expired - remove from cache
                    del self.metadata_cache[query_hash]
                    self.stats.misses += 1
                    return None

                # Cache hit - move to end (LRU)
                self.metadata_cache.move_to_end(query_hash)
                self.stats.hits += 1

                # Get embedding from appropriate cache
                if self.gpu_cache is not None:
                    embedding = self.gpu_cache[index].cpu().numpy()
                else:
                    embedding = self.cpu_cache[index].copy()

                return embedding
            else:
                self.stats.misses += 1
                return None

    def put(self, query: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        query_hash = self._hash_query(query)
        current_time = time.time()

        with self.lock:
            # Find available slot
            if len(self.metadata_cache) >= self.max_entries:
                # Evict oldest entry (LRU)
                oldest_hash = next(iter(self.metadata_cache))
                del self.metadata_cache[oldest_hash]
                self.stats.evictions += 1

            # Find next available index
            used_indices = {meta[0] for meta in self.metadata_cache.values()}
            index = 0
            while index in used_indices and index < self.max_entries:
                index += 1

            if index >= self.max_entries:
                print("[WARNING] Cache full, cannot store new embedding")
                return

            # Store in appropriate cache
            embedding_normalized = embedding.astype(np.float32)
            if self.gpu_cache is not None:
                self.gpu_cache[index] = torch.from_numpy(embedding_normalized).to(self.device)
            else:
                self.cpu_cache[index] = embedding_normalized

            # Update metadata
            self.metadata_cache[query_hash] = (index, current_time, query)

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        self.stats.total_size_mb = len(self.metadata_cache) * self.embedding_dim * 4 / (1024 * 1024)

        if self.use_cuda and self.gpu_cache is not None:
            self.stats.gpu_memory_mb = self.gpu_cache.element_size() * self.gpu_cache.nelement() / (1024 * 1024)

        return self.stats

    def clear(self):
        """Clear cache."""
        with self.lock:
            self.metadata_cache.clear()
            if self.gpu_cache is not None:
                self.gpu_cache.zero_()
            else:
                self.cpu_cache.fill(0)

            self.stats = CacheStats()
            print("🗑️  Cache cleared")


class CUDAResultCache:
    """Cache for complete RAG pipeline results."""

    def __init__(self, cache_dir="./cache", max_size_mb=100):
        """
        Initialize result cache.

        Args:
            cache_dir: Directory for persistent cache
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024

        # In-memory cache for recent results
        self.memory_cache = OrderedDict()  # {hash: (result, timestamp, size)}
        self.current_size = 0
        self.lock = threading.Lock()

        # Load existing cache index
        self.disk_cache_index = self._load_cache_index()

        print(f"💾 Result cache initialized: {cache_dir}, {max_size_mb}MB limit")

    def _hash_request(self, statement: str, model_config: Dict) -> str:
        """Generate hash for complete request."""
        request_str = f"{statement}|{model_config}"
        return hashlib.sha256(request_str.encode()).hexdigest()

    def _load_cache_index(self) -> Dict:
        """Load cache index from disk."""
        index_file = self.cache_dir / "cache_index.pkl"
        if index_file.exists():
            try:
                with open(index_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return {}

    def _save_cache_index(self):
        """Save cache index to disk."""
        index_file = self.cache_dir / "cache_index.pkl"
        try:
            with open(index_file, 'wb') as f:
                pickle.dump(self.disk_cache_index, f)
        except Exception as e:
            print(f"[WARNING] Failed to save cache index: {e}")

    def get(self, statement: str, model_config: Dict) -> Optional[Tuple[int, int]]:
        """Get cached result."""
        request_hash = self._hash_request(statement, model_config)
        current_time = time.time()

        # Check memory cache first
        with self.lock:
            if request_hash in self.memory_cache:
                result, timestamp, size = self.memory_cache[request_hash]

                # Check TTL (1 hour)
                if current_time - timestamp < 3600:
                    # Move to end (LRU)
                    self.memory_cache.move_to_end(request_hash)
                    return result
                else:
                    # Expired
                    del self.memory_cache[request_hash]
                    self.current_size -= size

        # Check disk cache
        if request_hash in self.disk_cache_index:
            cache_file = self.cache_dir / f"{request_hash}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)

                    result = cached_data['result']
                    timestamp = cached_data['timestamp']

                    # Check TTL (24 hours for disk cache)
                    if current_time - timestamp < 86400:
                        # Move to memory cache
                        self._add_to_memory_cache(request_hash, result, current_time)
                        return result
                    else:
                        # Expired - remove
                        cache_file.unlink()
                        del self.disk_cache_index[request_hash]

                except Exception as e:
                    print(f"[WARNING] Failed to load cached result: {e}")

        return None

    def put(self, statement: str, model_config: Dict, result: Tuple[int, int]) -> None:
        """Store result in cache."""
        request_hash = self._hash_request(statement, model_config)
        current_time = time.time()

        # Add to memory cache
        self._add_to_memory_cache(request_hash, result, current_time)

        # Add to disk cache
        cache_file = self.cache_dir / f"{request_hash}.pkl"
        try:
            cached_data = {
                'result': result,
                'timestamp': current_time,
                'statement': statement[:100],  # Truncated for debugging
                'model_config': model_config
            }

            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)

            self.disk_cache_index[request_hash] = {
                'timestamp': current_time,
                'file_size': cache_file.stat().st_size
            }

            # Cleanup old entries if needed
            self._cleanup_disk_cache()

        except Exception as e:
            print(f"[WARNING] Failed to cache result: {e}")

    def _add_to_memory_cache(self, request_hash: str, result: Tuple[int, int], timestamp: float):
        """Add result to memory cache with LRU eviction."""
        result_size = len(pickle.dumps(result))

        with self.lock:
            # Evict if needed
            while (self.current_size + result_size > self.max_size_bytes and
                   len(self.memory_cache) > 0):
                oldest_hash = next(iter(self.memory_cache))
                _, _, old_size = self.memory_cache[oldest_hash]
                del self.memory_cache[oldest_hash]
                self.current_size -= old_size

            # Add new entry
            self.memory_cache[request_hash] = (result, timestamp, result_size)
            self.current_size += result_size

    def _cleanup_disk_cache(self):
        """Remove old disk cache entries."""
        current_time = time.time()
        expired_hashes = []

        for request_hash, metadata in self.disk_cache_index.items():
            if current_time - metadata['timestamp'] > 86400:  # 24 hours
                expired_hashes.append(request_hash)

        for request_hash in expired_hashes:
            cache_file = self.cache_dir / f"{request_hash}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            del self.disk_cache_index[request_hash]

        if expired_hashes:
            self._save_cache_index()
            print(f"🗑️  Cleaned up {len(expired_hashes)} expired cache entries")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        disk_size = sum(meta['file_size'] for meta in self.disk_cache_index.values())

        return {
            'memory_entries': len(self.memory_cache),
            'memory_size_mb': self.current_size / (1024 * 1024),
            'disk_entries': len(self.disk_cache_index),
            'disk_size_mb': disk_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024)
        }

    def cleanup(self):
        """Clean up cache resources."""
        self._save_cache_index()
        with self.lock:
            self.memory_cache.clear()
            self.current_size = 0
        print("💾 Result cache cleaned up")


class CUDAOptimizedRAGPipeline:
    """RAG pipeline with comprehensive CUDA caching optimizations."""

    def __init__(self, base_pipeline, cache_config: Dict = None):
        """
        Initialize CUDA-optimized pipeline with caching.

        Args:
            base_pipeline: Base RAG pipeline to wrap
            cache_config: Caching configuration
        """
        self.base_pipeline = base_pipeline

        # Default cache configuration
        default_config = {
            'embedding_cache_mb': 256,
            'result_cache_mb': 50,
            'cache_dir': './cache',
            'enable_gpu_cache': True
        }

        cache_config = cache_config or {}
        self.cache_config = {**default_config, **cache_config}

        # Initialize caches
        embedding_dim = getattr(base_pipeline.document_store.embedding_model, 'get_dimension', lambda: 768)()

        self.embedding_cache = CUDAEmbeddingCache(
            cache_size_mb=self.cache_config['embedding_cache_mb'],
            embedding_dim=embedding_dim,
            device=base_pipeline.document_store.device
        )

        self.result_cache = CUDAResultCache(
            cache_dir=self.cache_config['cache_dir'],
            max_size_mb=self.cache_config['result_cache_mb']
        )

        # Model configuration for cache keys
        self.model_config = {
            'embedding_model': base_pipeline.embedding_model,
            'llm_model': base_pipeline.llm_client.model_name,
            'top_k': base_pipeline.top_k
        }

        print("⚡ CUDA-optimized pipeline with caching initialized")

    def predict(self, statement: str) -> Tuple[int, int]:
        """Make prediction with caching."""
        # Check result cache first
        cached_result = self.result_cache.get(statement, self.model_config)
        if cached_result is not None:
            print("💨 Cache hit - returning cached result")
            return cached_result

        # Check embedding cache
        cached_embedding = self.embedding_cache.get(statement)
        if cached_embedding is not None:
            print("⚡ Embedding cache hit")
            # TODO: Implement search with cached embedding
            # For now, fallback to full prediction

        # Full prediction
        result = self.base_pipeline.predict(statement)

        # Cache the result
        self.result_cache.put(statement, self.model_config, result)

        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        embedding_stats = self.embedding_cache.get_stats()
        result_stats = self.result_cache.get_cache_stats()

        return {
            'embedding_cache': {
                'hit_rate': embedding_stats.hit_rate,
                'hits': embedding_stats.hits,
                'misses': embedding_stats.misses,
                'size_mb': embedding_stats.total_size_mb
            },
            'result_cache': result_stats
        }

    def cleanup(self):
        """Clean up all cache resources."""
        self.embedding_cache.memory_manager.cleanup()
        self.result_cache.cleanup()
        print("⚡ CUDA-optimized pipeline cleaned up")
