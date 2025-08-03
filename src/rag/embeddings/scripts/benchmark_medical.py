#!/usr/bin/env python3
"""Benchmark embedding models on medical emergency statements."""

import sys
from pathlib import Path
import time

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from rag.embeddings.models import get_embedding_model


def get_medical_statements():
    """Get a comprehensive set of medical emergency statements."""
    return [
        # Cardiovascular emergencies
        "Acute ST-elevation myocardial infarction with cardiogenic shock requiring immediate PCI",
        "Unstable angina with dynamic ECG changes and elevated troponin levels",
        "Acute pulmonary edema secondary to decompensated heart failure",
        "Hypertensive crisis with end-organ damage",
        # Neurological emergencies
        "Acute ischemic stroke within thrombolysis window",
        "Intracerebral hemorrhage with midline shift requiring urgent neurosurgical intervention",
        "Status epilepticus refractory to benzodiazepines",
        "Bacterial meningitis with altered mental status and nuchal rigidity",
        # Respiratory emergencies
        "Tension pneumothorax requiring immediate needle decompression",
        "Massive pulmonary embolism with hemodynamic instability",
        "Severe asthma exacerbation with impending respiratory failure",
        "Acute respiratory distress syndrome requiring mechanical ventilation",
        # Trauma
        "Multiple trauma with unstable pelvic fracture and hemorrhagic shock",
        "Traumatic brain injury with GCS 8 requiring intubation",
        "Penetrating chest trauma with cardiac tamponade",
        "Spinal cord injury with neurogenic shock",
        # Metabolic emergencies
        "Diabetic ketoacidosis with severe metabolic acidosis",
        "Hyperosmolar hyperglycemic state with altered sensorium",
        "Severe hypoglycemia with seizures",
        "Thyroid storm with multi-organ dysfunction",
        # Other critical conditions
        "Anaphylactic shock requiring epinephrine and airway management",
        "Septic shock with multi-organ failure",
        "Acute abdomen with signs of peritonitis",
        "Testicular torsion requiring emergent surgical detorsion",
    ]


def benchmark_model(model_name, statements):
    """Benchmark a single model."""
    results = {
        "model_name": model_name,
        "load_time": 0,
        "encode_time": 0,
        "dimension": 0,
        "memory_estimate": 0,
        "error": None,
    }

    try:
        # Load model
        start = time.time()
        model = get_embedding_model(model_name)
        results["load_time"] = time.time() - start

        # Encode statements
        start = time.time()
        embeddings = model.encode(statements, show_progress_bar=False)
        results["encode_time"] = time.time() - start

        # Get statistics
        results["dimension"] = embeddings.shape[1]
        results["memory_estimate"] = embeddings.nbytes / (1024 * 1024)  # MB

        # Clean up
        del model
        del embeddings

    except Exception as e:
        results["error"] = str(e)

    return results


def main():
    """Run benchmarks."""
    print("=" * 80)
    print("MEDICAL EMBEDDING MODEL BENCHMARK")
    print("=" * 80)

    # Get medical statements
    statements = get_medical_statements()
    print(f"\nBenchmarking on {len(statements)} medical emergency statements")

    # Models to benchmark
    models_to_test = [
        # Small models (fast, lower quality)
        "all-MiniLM-L6-v2",  # 384 dim, 22M params
        # Medium models (balanced)
        "all-mpnet-base-v2",  # 768 dim, 110M params
        "gte-base",  # 768 dim
        "bge-base-en-v1.5",  # 768 dim
        # Large models (slower, higher quality) - commented out for quick test
        # "gte-large",  # 1024 dim
        # "bge-large-en-v1.5",  # 1024 dim
        # Medical-specific (if available)
        # "pubmedbert-base-embeddings",  # 768 dim, medical-focused
    ]

    results = []

    print("\nRunning benchmarks...")
    print("-" * 80)

    for model_name in models_to_test:
        print(f"\nTesting {model_name}...", end=" ", flush=True)
        result = benchmark_model(model_name, statements)
        results.append(result)

        if result["error"]:
            print(f"ERROR: {result['error']}")
        else:
            print(f"✓ ({result['encode_time']:.2f}s)")

    # Display results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Sort by encoding speed
    successful_results = [r for r in results if not r["error"]]
    successful_results.sort(key=lambda x: x["encode_time"])

    print("\nModel Performance Summary:")
    print("-" * 80)
    print(
        f"{'Model':<25} {'Dim':<6} {'Load(s)':<8} {'Encode(s)':<10} {'ms/stmt':<10} {'Memory(MB)':<10}"
    )
    print("-" * 80)

    for r in successful_results:
        ms_per_stmt = (r["encode_time"] / len(statements)) * 1000
        print(
            f"{r['model_name']:<25} {r['dimension']:<6} {r['load_time']:<8.2f} "
            f"{r['encode_time']:<10.3f} {ms_per_stmt:<10.1f} {r['memory_estimate']:<10.2f}"
        )

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR YOUR USE CASE")
    print("=" * 80)

    print("\n1. For SPEED (real-time emergency response):")
    print("   - all-MiniLM-L6-v2: Fastest encoding, good for quick retrieval")
    print("   - Dimension: 384 (smaller index, faster search)")

    print("\n2. For QUALITY (accuracy is critical):")
    print("   - bge-base-en-v1.5: Better semantic understanding")
    print("   - all-mpnet-base-v2: Well-balanced performance")
    print("   - Dimension: 768 (better representation capacity)")

    print("\n3. For MEDICAL DOMAIN:")
    print("   - Consider fine-tuning chosen model on medical data")
    print("   - Use medical preprocessing for abbreviations")
    print("   - Test with your actual medical documents")

    print("\n4. DEPLOYMENT CONSIDERATIONS:")
    print("   - Memory: ~100-200MB per model")
    print("   - CPU inference is viable for all tested models")
    print("   - Consider caching embeddings for common queries")


if __name__ == "__main__":
    main()
