from pathlib import Path
import json
from typing import List, Dict, Tuple

# === Configuration ===
REPHRASE_COUNT = 3  # Number of rephrased versions per statement


# === Load statements and answers from files ===
def load_statements_and_answers(
    statement_dir: str, answer_dir: str
) -> List[Dict[str, str]]:
    """
    Load paired statement (.txt) and answer (.json) files.
    Assumes filenames (without extension) are aligned, e.g., stmt_001.txt <-> stmt_001.json

    Returns:
        List of dicts: [
            {
                "statement": "High BP causes stroke...",
                "output": '{"statement_is_true": 1, "statement_topic": 42}'
            },
            ...
        ]
    """
    statement_path = Path(statement_dir)
    answer_path = Path(answer_dir)

    if not statement_path.exists():
        raise FileNotFoundError(f"Statement directory not found: {statement_dir}")
    if not answer_path.exists():
        raise FileNotFoundError(f"Answer directory not found: {answer_dir}")

    # Get sorted .txt and .json files
    statement_files = sorted(
        [f for f in statement_path.iterdir() if f.suffix == ".txt"]
    )
    answer_files = sorted([f for f in answer_path.iterdir() if f.suffix == ".json"])

    if len(statement_files) != len(answer_files):
        raise ValueError("Mismatch in number of statement and answer files.")

    data = []
    for stmt_file, ans_file in zip(statement_files, answer_files):
        # Ensure base names match
        if stmt_file.stem != ans_file.stem:
            print(f"Warning: Mismatched pair: {stmt_file.name} != {ans_file.name}")

        # Read statement
        with open(stmt_file, "r", encoding="utf-8") as f:
            statement_text = f.read().strip()

        # Read answer (assumed JSON)
        with open(ans_file, "r", encoding="utf-8") as f:
            answer_json = json.load(f)
            # Convert back to string to preserve format (or process as needed)
            answer_str = json.dumps(answer_json)

        data.append({"statement": statement_text, "output": answer_str})

    return data


# === Rephrase statement using dummy logic (replace with real model/API later) ===
def rephrase_statement(
    statement: str, original_output: str, rephrases: int = 3
) -> List[Tuple[str, str]]:
    """
    Generate multiple rephrased versions of a statement.
    For now, uses simple placeholder logic. Replace with NLP model or API call.

    Args:
        statement (str): Original statement.
        original_output (str): Corresponding label/output (e.g., JSON string).
        rephrases (int): Number of rephrased variants to generate.

    Returns:
        List[Tuple[str, str]]: List of (rephrased_statement, output)
    """
    # TODO: Integrate with Hugging Face, OpenAI, etc.
    # Example using simple templates
    templates = [
        f"In other words, {statement.lower()}",
        f"It is {'true' if '1' in original_output else 'false'} that {statement[0].lower() + statement[1:]}",
        f"One could say that {statement.lower()}",
    ]

    # Truncate or repeat if needed
    rephrased = []
    for i in range(rephrases):
        rephrased.append((templates[i % len(templates)], original_output))

    return rephrased


# === Write rephrased statements and answers to output directory ===
def write_statements_to_folder(data: List[Dict[str, str]], output_dir: Path):
    """
    Write rephrased statements and their answers to output directory.
    Creates:
        output_dir/statements/stmt_001.txt
        output_dir/answers/ans_001.json
        ... plus rephrased versions

    Args:
        data (List[Dict]): List of original statement-answer pairs.
        output_dir (Path): Root output directory.
    """
    statement_out_dir = output_dir / "statements"
    answer_out_dir = output_dir / "answers"

    statement_out_dir.mkdir(parents=True, exist_ok=True)
    answer_out_dir.mkdir(parents=True, exist_ok=True)

    index = 0
    for item in data:
        statement = item["statement"]
        output = item["output"]

        # Generate rephrased versions
        rephrased_pairs = rephrase_statement(
            statement, output, rephrases=REPHRASE_COUNT
        )

        for rephrased_text, out_str in rephrased_pairs:
            # Write rephrased statement
            stmt_file = statement_out_dir / f"stmt_{index:03d}.txt"
            with open(stmt_file, "w", encoding="utf-8") as f:
                f.write(rephrased_text)

            # Write corresponding answer
            ans_file = answer_out_dir / f"ans_{index:03d}.json"
            with open(ans_file, "w", encoding="utf-8") as f:
                f.write(out_str)

            index += 1

    print(f"Generated {index} rephrased statement-answer pairs in {output_dir}")


# === Main execution ===
if __name__ == "__main__":
    # Input directories
    statement_dir = "rag/data/raw/train/statements"
    answer_dir = "rag/data/raw/train/answers"

    # Output directory
    output_dir = Path("rag/data/rephrased/train")

    # Load data
    print("Loading statements and answers...")
    data = load_statements_and_answers(statement_dir, answer_dir)

    if not data:
        raise ValueError("No data loaded. Check input directories.")

    print(f"Loaded {len(data)} statement-answer pairs.")

    # Generate and write rephrased data
    print("Generating rephrased statements...")
    write_statements_to_folder(data, output_dir)

    print("Done.")
