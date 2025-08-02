"""
Validation wrapper for Tumor Segmentation.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from shared.validation.segmentation_validate import main

if __name__ == "__main__":
    main()
