#!/usr/bin/env python3
"""Download and cache embedding models."""

import argparse
from pathlib import Path
import sys
from tqdm import tqdm

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from rag.embeddings.managers import ModelManager
from rag.embeddings.models import get_registry


def download_models(model_names=None, all_models=False):
    """Download specified models or all models."""

    # Initialize manager
    manager = ModelManager()
    registry = get_registry()

    # Determine which models to download
    if all_models:
        models_to_download = registry.list_models()
    elif model_names:
        models_to_download = model_names
    else:
        # Default models
        models_to_download = [
            "all-MiniLM-L6-v2",
            "nomic-embed-text-v1.5",
            "gte-base",
            "bge-base-en-v1.5",
        ]

    print(f"Downloading {len(models_to_download)} models...")

    # Download each model
    for model_name in tqdm(models_to_download, desc="Downloading models"):
        try:
            print(f"\nDownloading {model_name}...")
            model = manager.load_model(model_name)

            # Warm up the model
            model.warmup()

            # Get model info
            info = {
                "dimension": model.get_dimension(),
                "max_seq_length": model.get_max_seq_length(),
                "supports_matryoshka": model.supports_matryoshka(),
            }

            print(f"✓ {model_name} downloaded successfully")
            print(f"  Dimension: {info['dimension']}")
            print(f"  Max sequence length: {info['max_seq_length']}")
            if info["supports_matryoshka"]:
                print(f"  Matryoshka dimensions: {model.get_matryoshka_dimensions()}")

        except Exception as e:
            print(f"✗ Error downloading {model_name}: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("Download Summary:")
    cached_models = manager.list_cached_models()
    print(f"Successfully cached {len(cached_models)} models:")
    for model in cached_models:
        print(f"  - {model}")


def main():
    parser = argparse.ArgumentParser(description="Download embedding models")
    parser.add_argument("--models", nargs="+", help="Specific models to download")
    parser.add_argument(
        "--all", action="store_true", help="Download all available models"
    )
    parser.add_argument(
        "--medical", action="store_true", help="Download medical-specific models"
    )

    args = parser.parse_args()

    if args.medical:
        models = ["pubmedbert-base-embeddings", "BioLORD-2023"]
    else:
        models = args.models

    download_models(models, args.all)


if __name__ == "__main__":
    main()
