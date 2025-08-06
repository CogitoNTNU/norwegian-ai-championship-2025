#!/usr/bin/env python3
"""
Test script to verify local model support and backward compatibility.
"""

import subprocess
import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"


def run_test(description, command):
    """Run a test command and report results."""
    print(f"\n{'=' * 60}")
    print(f"TEST: {description}")
    print(f"Command: {' '.join(command)}")
    print("-" * 60)

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)

        if "Error" in result.stderr or result.returncode != 0:
            print("X Test failed!")
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print(f"Error output:\n{result.stderr[:500]}")
            return False
        else:
            print("OK Test passed!")
            # Show relevant output lines
            for line in result.stdout.split("\n"):
                if any(
                    keyword in line
                    for keyword in [
                        "Statements:",
                        "Data Path:",
                        "Embedding:",
                        "LLM:",
                        "Strategy:",
                        "Using",
                        "Found",
                        "Loading",
                    ]
                ):
                    print(f"  {line}")
            return True
    except subprocess.TimeoutExpired:
        print("TIMEOUT Test timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"ERROR Test error: {e}")
        return False


def main():
    print("=" * 60)
    print("TESTING LOCAL MODEL SUPPORT AND BACKWARD COMPATIBILITY")
    print("=" * 60)

    script_path = Path(__file__).parent / "test_pipeline.py"
    python_cmd = sys.executable

    tests_passed = 0
    tests_total = 0

    # Test 1: Backward compatibility - default behavior
    tests_total += 1
    if run_test(
        "Backward compatibility - default behavior (no new flags)",
        [python_cmd, str(script_path), "--n", "1", "--verbose"],
    ):
        tests_passed += 1

    # Test 2: Custom data path
    tests_total += 1
    if run_test(
        "Custom data path",
        [python_cmd, str(script_path), "--n", "1", "--data-path", "data", "--verbose"],
    ):
        tests_passed += 1

    # Test 3: Models path that doesn't exist (should fall back to registry)
    tests_total += 1
    if run_test(
        "Non-existent models path (should show warning and fall back)",
        [
            python_cmd,
            str(script_path),
            "--n",
            "1",
            "--models-path",
            "nonexistent_models",
            "--embedding",
            "all-MiniLM-L6-v2",
        ],
    ):
        tests_passed += 1

    # Test 4: Local model path that doesn't exist (should fail gracefully)
    tests_total += 1
    print("\n" + "=" * 60)
    print("TEST: Local model path that doesn't exist (expected to fail)")
    print("Command: python test_pipeline.py --n 1 --local-model nonexistent_model")
    print("-" * 60)
    try:
        result = subprocess.run(
            [
                python_cmd,
                str(script_path),
                "--n",
                "1",
                "--local-model",
                "nonexistent_model",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "Local model path does not exist" in result.stdout:
            print("OK Test passed - correctly detected missing model")
            tests_passed += 1
        else:
            print("X Test failed - didn't properly handle missing model")
    except Exception as e:
        print(f"ERROR Test error: {e}")

    # Test 5: Invalid data path (should fail gracefully)
    tests_total += 1
    print("\n" + "=" * 60)
    print("TEST: Invalid data path (expected to fail)")
    print("Command: python test_pipeline.py --n 1 --data-path nonexistent_data")
    print("-" * 60)
    try:
        result = subprocess.run(
            [
                python_cmd,
                str(script_path),
                "--n",
                "1",
                "--data-path",
                "nonexistent_data",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if (
            "Data path does not exist" in result.stdout
            or "Training data not found" in result.stdout
        ):
            print("OK Test passed - correctly detected missing data")
            tests_passed += 1
        else:
            print("X Test failed - didn't properly handle missing data")
    except Exception as e:
        print(f"ERROR Test error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{tests_total}")

    if tests_passed == tests_total:
        print("SUCCESS! All tests passed! The implementation is working correctly.")
    else:
        print(
            f"WARNING: {tests_total - tests_passed} test(s) failed. Please review the implementation."
        )

    return 0 if tests_passed == tests_total else 1


if __name__ == "__main__":
    sys.exit(main())
