"""
Validation wrapper for Emergency Healthcare RAG.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from shared.validation.rag_validate import main

if __name__ == "__main__":
    main()
