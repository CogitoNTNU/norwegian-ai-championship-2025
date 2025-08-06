import asyncio
from pathlib import Path
import json
import re
from typing import List, Dict, Optional, Tuple
from ollama import AsyncClient, ChatResponse

# === Configuration ===
REPHRASE_COUNT = 3  # Number of rephrased versions per statement


# === API Client Setup ===
def get_api_key(file_path=".api_key") -> str:
    try:
        with open(file_path, "r") as f:
            api_key = f.read().strip()
        return api_key
    except FileNotFoundError:
        print(f"API key file not found: {file_path}")
        raise


# -- Get the API key from a file outside the git repo --
api_key_file = Path.cwd().parent / ".api_key"
api_key = get_api_key(api_key_file)

client = AsyncClient(
    host="https://beta.chat.nhn.no/ollama",
    headers={"Authorization": f"{api_key}"},
)


# -- Function to call LLM-api --
async def call_llm_api(
    prompt: str, model: str, num_rephrases: int = 3
) -> Optional[List[str]]:
    """
    Asks the LLM to rephrase a statement in multiple ways.

    Expects response in format:
        1. Rephrased statement one.
        2. Another version of the statement.
        3. A third way to say it.

    Args:
        prompt (str): The original statement to rephrase.
        model (str): The model identifier (e.g., "llama3-8b").
        num_rephrases (int): Number of rephrased versions expected (default: 3).

    Returns:
        List[str]: List of rephrased statements, or None if failed.
    """
    system_message = (
        f"Rephrase the following statement in exactly {num_rephrases} distinct ways. "
        "Preserve the original meaning and truthfulness. "
        "Respond ONLY in this format:\n"
        "1. #first rephrased statement\n"
        "2. #second rephrased statement\n"
        "3. #third rephrased statement\n"
        "Do not include any other text, explanations, or numbering beyond this."
    )

    try:
        # Call the LLM
        response: ChatResponse = await asyncio.wait_for(
            client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
            ),
            timeout=60,  # Set a reasonable timeout for the API call
        )

        # Extract content
        if not hasattr(response, "message") or not hasattr(response.message, "content"):
            print("Invalid response structure: missing message or content.")
            return None

        content = response.message.content.strip()
        if not content:
            print("Empty response from LLM.")
            return None

        # Regex pattern to match numbered lines: "1. ...", "2. ...", etc.
        pattern = r"^\s*\d+\.\s+(.+?)(?=(?:\n\s*\d+\.|\Z))"
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

        print(matches)

        # Clean up matches (remove trailing whitespace, newlines)
        rephrased = [match.strip() for match in matches]

        # Validate number of rephrased statements
        if len(rephrased) < num_rephrases:
            print(
                f"Warning: Expected {num_rephrases} rephrased statements, got {len(rephrased)}. Raw response:\n{content}"
            )
        elif len(rephrased) > num_rephrases:
            rephrased = rephrased[:num_rephrases]  # Truncate if too many

        return rephrased

    except asyncio.TimeoutError:
        print("LLM API call timed out.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while calling the LLM: {e}")
        return None


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


def filter_diverse_responses(
    statements: List[str], min_similarity_threshold: float = 0.8
) -> List[str]:
    """
    Filters out rephrased statements that are too similar to each other,
    keeping only diverse ones.

    Uses token-based Jaccard similarity as a lightweight diversity check.

    Args:
        statements (List[str]): List of rephrased statement strings.
        min_similarity_threshold (float): If similarity >= this value, statements are considered too similar.
                                         Lower = stricter diversity (default: 0.8)

    Returns:
        List[str]: Deduplicated and diverse list of statements.
    """

    def jaccard_similarity(s1: str, s2: str) -> float:
        set1 = set(s1.lower().split())
        set2 = set(s2.lower().split())
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = set1 & set2
        union = set1 | set2
        return len(intersection) / len(union)

    diverse = []
    for stmt in statements:
        # Skip empty or whitespace-only
        if not stmt.strip():
            continue

        # Check similarity with already accepted diverse statements
        is_too_similar = False
        for existing in diverse:
            if jaccard_similarity(stmt, existing) >= min_similarity_threshold:
                is_too_similar = True
                break

        if not is_too_similar:
            diverse.append(stmt)

    return diverse


# === Rephrase statement using dummy logic (replace with real model/API later) ===
# Assuming your async LLM function is defined as:
# async def call_llm_api(prompt: str, model: str, num_rephrases: int = 3) -> List[str] | None:


async def rephrase_statement(
    statement: str, original_output: str, rephrases: int = 3
) -> List[Tuple[str, str]]:
    """
    Generate multiple rephrased versions of a statement using an LLM,
    and pair each with the original output.

    Args:
        statement (str): Original statement to rephrase.
        original_output (str): Ground truth label/output (e.g., JSON string).
        rephrases (int): Number of rephrased variants to generate.

    Returns:
        List[Tuple[str, str]]: List of (rephrased_statement, original_output)
    """
    # Call the async LLM API
    rephrased_list = await call_llm_api(
        prompt=statement, model="nhn-small:latest", num_rephrases=rephrases
    )

    # Handle failure or empty response
    if not rephrased_list:
        print(
            f"Failed to rephrase statement: '{statement}'. Using original as fallback."
        )
        # Fallback: repeat the original statement
        rephrased_list = [statement] * rephrases

    # Truncate or pad to ensure exactly `rephrases` outputs
    while len(rephrased_list) < rephrases:
        rephrased_list.append(statement)  # pad with original if needed
    rephrased_list = rephrased_list[:rephrases]  # truncate if too many

    # Pair each rephrased statement with the same original output
    return [(rephrased_stmt, original_output) for rephrased_stmt in rephrased_list]


# === Write rephrased statements and answers to output directory ===
async def write_statements_to_folder(data: List[Dict[str, str]], output_dir: Path):
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
        rephrased_pairs = await rephrase_statement(
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
    asyncio.run(write_statements_to_folder(data, output_dir))

    print("Done.")
