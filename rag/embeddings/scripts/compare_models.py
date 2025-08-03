#!/usr/bin/env python3
"""Compare different embedding models on medical statements."""

import sys
from pathlib import Path
import time
import numpy as np
from typing import List, Dict
import argparse

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from rag.embeddings.models import get_embedding_model, get_registry


def get_test_statements() -> List[str]:
    """Get test medical statements."""
    return [
        "Patient presents with acute chest pain radiating to left arm, diaphoresis noted",
        "Testicular torsion requires immediate surgical intervention to prevent necrosis",
        "ECG shows ST-segment elevation in leads II, III, and aVF suggesting inferior MI",
        "Anaphylactic shock: administer epinephrine 0.3mg IM immediately",
        "Glasgow Coma Scale: Eye opening 3, Verbal response 4, Motor response 5, Total GCS 12",
        "Suspected pulmonary embolism, initiate anticoagulation pending CT angiography",
        "Diabetic ketoacidosis: pH 7.1, glucose 450mg/dL, positive ketones",
        "Acute appendicitis with peritoneal signs, prepare for emergency appendectomy",
        "Status epilepticus unresponsive to lorazepam, loading phenytoin 20mg/kg",
        "Tension pneumothorax identified, immediate needle decompression required",
    ]


def compute_similarities(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarities between embeddings."""
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms

    # Compute similarities
    similarities = np.dot(normalized, normalized.T)
    return similarities


def compare_models(
    model_names: List[str], statements: List[str], verbose: bool = True
) -> Dict:
    """Compare multiple models."""
    results = {}

    for model_name in model_names:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Testing {model_name}")
            print("=" * 60)

        try:
            # Load model
            start_time = time.time()
            model = get_embedding_model(model_name)
            load_time = time.time() - start_time

            # Encode statements
            start_time = time.time()
            embeddings = model.encode(statements, show_progress_bar=False)
            encode_time = time.time() - start_time

            # Compute statistics
            dimension = embeddings.shape[1]
            similarities = compute_similarities(embeddings)
            avg_similarity = np.mean(
                similarities[np.triu_indices_from(similarities, k=1)]
            )

            results[model_name] = {
                "dimension": dimension,
                "load_time": load_time,
                "encode_time": encode_time,
                "avg_similarity": avg_similarity,
                "embeddings": embeddings,
                "similarities": similarities,
            }

            if verbose:
                print(f"✓ Dimension: {dimension}")
                print(f"✓ Load time: {load_time:.2f}s")
                print(
                    f"✓ Encode time: {encode_time:.3f}s ({encode_time / len(statements) * 1000:.1f}ms per statement)"
                )
                print(f"✓ Average similarity: {avg_similarity:.3f}")

                if model.supports_matryoshka():
                    dims = model.get_matryoshka_dimensions()
                    print(f"✓ Matryoshka dimensions: {dims}")

        except Exception as e:
            if verbose:
                print(f"✗ Error: {e}")
            results[model_name] = {"error": str(e)}

    return results


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description="Compare embedding models")
    parser.add_argument("--models", nargs="+", help="Models to compare")
    parser.add_argument("--all", action="store_true", help="Test all available models")
    parser.add_argument(
        "--quick", action="store_true", help="Quick test with fewer statements"
    )
    args = parser.parse_args()

    # Get models to test
    registry = get_registry()
    if args.all:
        model_names = registry.list_models()
    elif args.models:
        model_names = args.models
    else:
        # Default models
        model_names = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "bge-base-en-v1.5",
        ]

    # Get test statements
    statements = get_test_statements()
    if args.quick:
        statements = statements[:3]  # Use only first 3 for quick test

    print("=" * 60)
    print("EMBEDDING MODEL COMPARISON")
    print("=" * 60)
    print(f"Testing {len(model_names)} models on {len(statements)} statements")

    # Compare models
    results = compare_models(model_names, statements)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful_models = [m for m in results if "error" not in results[m]]
    if successful_models:
        print(f"\nSuccessfully tested {len(successful_models)} models:")
        for model_name in successful_models:
            r = results[model_name]
            print(f"\n{model_name}:")
            print(f"  - Dimension: {r['dimension']}")
            print(
                f"  - Encode speed: {r['encode_time'] / len(statements) * 1000:.1f}ms per statement"
            )
            print(f"  - Average similarity: {r['avg_similarity']:.3f}")

    failed_models = [m for m in results if "error" in results[m]]
    if failed_models:
        print(f"\nFailed to test {len(failed_models)} models:")
        for model_name in failed_models:
            print(f"  - {model_name}: {results[model_name]['error']}")


if __name__ == "__main__":
    main()
