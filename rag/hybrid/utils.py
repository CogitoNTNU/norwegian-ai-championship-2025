"""
Utility functions for hybrid RAG pipeline
"""

from contextlib import contextmanager
import time
import logging

logger = logging.getLogger(__name__)

@contextmanager
def stopwatch(label: str):
    """Context manager for measuring execution time of code blocks."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    logger.info("%-18s : %.3f s", label, dt)
