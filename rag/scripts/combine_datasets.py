#!/usr/bin/env python3
"""
Combine all raw training samples and synthetic samples into one unified dataset
for model training, leaving originals untouched.
"""

import argparse
from pathlib import Path
import shutil

# Define source and target directories
RAW_STATEMENTS_DIR = Path("../data/raw/train/statements/")
RAW_ANSWERS_DIR = Path("../data/raw/train/answers/")
SYNTH_TRUE_DIR = Path("../data/processed/syntetic_true/")
SYNTH_FALSE_DIR = Path("../data/processed/syntetic_false/")
COMBINED_DIR = Path("../data/processed/combined_train/")


def copy_files(source_dir, dest_dir, file_type):
    """Copy files from source to destination directory and return count."""
    count = 0
    if not source_dir.exists():
        print(f"Warning: Source directory {source_dir} does not exist")
        return count

    for file in source_dir.rglob(f"*.{file_type}"):
        dest_file = dest_dir / file.name
        if dest_file.exists():
            raise FileExistsError(f"File already exists: {dest_file}")
        shutil.copy2(file, dest_file)
        count += 1
    return count


def combine_datasets(force=False):
    """Main function to combine all datasets."""
    # Create target directories
    statements_dir = COMBINED_DIR / "statements"
    answers_dir = COMBINED_DIR / "answers"

    if COMBINED_DIR.exists():
        if not force:
            answer = input(
                "Combined dataset already exists. Delete and rebuild? [y/N] "
            )
            if answer.lower() != "y":
                print("Operation cancelled.")
                return
        shutil.rmtree(COMBINED_DIR)

    statements_dir.mkdir(parents=True, exist_ok=True)
    answers_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Copy raw files
        print("Copying raw training files...")
        n_raw_txt = copy_files(RAW_STATEMENTS_DIR, statements_dir, "txt")
        n_raw_json = copy_files(RAW_ANSWERS_DIR, answers_dir, "json")

        # Copy synthetic files
        print("Copying synthetic true files...")
        n_true_txt = copy_files(SYNTH_TRUE_DIR, statements_dir, "txt")
        n_true_json = copy_files(SYNTH_TRUE_DIR, answers_dir, "json")

        print("Copying synthetic false files...")
        n_false_txt = copy_files(SYNTH_FALSE_DIR, statements_dir, "txt")
        n_false_json = copy_files(SYNTH_FALSE_DIR, answers_dir, "json")

        # Console summary
        print(f"\n Combined dataset created at {COMBINED_DIR}")
        print(f"  - Raw samples copied     : {n_raw_txt}  (should be 200)")
        print(f"  - Synthetic samples copied: {n_true_txt} true + {n_false_txt} false")
        print(f"  - Total .txt files        : {n_raw_txt + n_true_txt + n_false_txt}")
        print(
            f"  - Total .json files       : {n_raw_json + n_true_json + n_false_json}"
        )

    except FileExistsError as e:
        print(f"Error: {e}")
        print("This indicates duplicate IDs between datasets. Please check your data.")
        return False
    except Exception as e:
        print(f"Error during dataset combination: {e}")
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine raw and synthetic datasets for model training."
    )
    parser.add_argument(
        "--force", action="store_true", help="Recreate combined dataset if it exists."
    )
    args = parser.parse_args()

    success = combine_datasets(force=args.force)
    if not success:
        exit(1)
