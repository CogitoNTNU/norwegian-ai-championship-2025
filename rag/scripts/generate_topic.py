#!/usr/bin/env python3
"""
Topic generation script for Emergency Healthcare RAG dataset.
Generates synthetic true/false statements for topics based on reference articles.
"""

import os
import json
import glob
import argparse
from typing import Dict, List

import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# Load environment variables from repo root
load_dotenv(dotenv_path="../../.env")

# Constants
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Initialize clients
client = anthropic.Anthropic(api_key=API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Topic ID to topic name mapping (from CSV)
TOPIC_MAPPING = {
    0: "Abdominal Trauma",
    1: "Acute Abdomen",
    2: "Acute Appendicitis",
    3: "Acute Cholecystitis",
    4: "Acute Coronary Syndrome",
    5: "Acute Kidney Injury",
    6: "Acute Liver Failure",
    7: "Acute Myocardial Infarction (STEMI_NSTEMI)",
    8: "Acute Respiratory Distress Syndrome",
    9: "Anaphylaxis",
    10: "Aortic Dissection",
    11: "Aortic Stenosis",
    12: "Arrhythmias (various)",
    13: "Aspiration Pneumonia",
    14: "Asthma Exacerbation",
    15: "Atrial Fibrillation",
    16: "Blunt Trauma",
    17: "Bowel Obstruction",
    18: "Brain Death",
    19: "Bronchitis",
    20: "Burns",
    21: "COPD Exacerbation",
    22: "Cardiac Arrest",
    23: "Cardiac Contusion",
    24: "Cardiac Tamponade",
    25: "Cardiomyopathy",
    26: "Cervical Spine Injury",
    27: "Chest Pain (non-cardiac)",
    28: "Compartment Syndrome",
    29: "Delirium",
    30: "Diabetic Ketoacidosis",
    31: "Eclampsia",
    32: "Ectopic Pregnancy",
    33: "Embolism",
    34: "Empyema",
    35: "Encephalitis",
    36: "Endocarditis",
    37: "GI Bleeding",
    38: "Heart Failure (Acute_Chronic)",
    39: "Hemothorax",
    40: "Hypertensive Emergency",
    41: "Hyperventilation Syndrome",
    42: "Hypoglycemia",
    43: "Hyponatremia_Hypernatremia",
    44: "Hypothermia_Hyperthermia",
    45: "Interstitial Lung Disease",
    46: "Lung Abscess",
    47: "Lung Cancer",
    48: "Meningitis",
    49: "Mitral Regurgitation",
    50: "Multi-organ Failure",
    51: "Myocarditis",
    52: "Ovarian Torsion",
    53: "Overdose_Poisoning",
    54: "Pancreatitis",
    55: "Penetrating Trauma",
    56: "Perforated Viscus",
    57: "Pericarditis",
    58: "Placental Abruption",
    59: "Pleural Effusion",
    60: "Pneumomediastinum",
    61: "Pneumonia (bacterial_viral_atypical)",
    62: "Pneumothorax",
    63: "Pulmonary Embolism",
    64: "Pulmonary Fibrosis",
    65: "Pulmonary Hypertension",
    66: "Respiratory Acidosis",
    67: "Respiratory Failure",
    68: "Rhabdomyolysis",
    69: "Right Heart Failure",
    70: "Ruptured AAA",
    71: "Seizures_Status Epilepticus",
    72: "Sepsis_Septic Shock",
    73: "Shock (various types)",
    74: "Sleep Apnea",
    75: "Stroke (Ischemic_Hemorrhagic)",
    76: "Syncope",
    77: "Takotsubo Cardiomyopathy",
    78: "Testicular Torsion",
    79: "Traumatic Brain Injury",
    80: "Unstable Angina",
    81: "Upper Airway Obstruction",
    82: "Ventricular Tachycardia",
    83: "12-lead ECG",
    84: "Angiography (invasive)",
    85: "Arterial Blood Gas (ABG)",
    86: "B-type Natriuretic Peptide (BNP)",
    87: "Blood Cultures",
    88: "Bronchoscopy with BAL",
    89: "C-Reactive Protein (CRP)",
    90: "CT Angiogram",
    91: "CT Other",
    92: "Cardiac Catheterization",
    93: "Central Venous Pressure",
    94: "Chest X-ray",
    95: "Coagulation Studies (PT_PTT_INR)",
    96: "Complete Blood Count (CBC) with differential",
    97: "Creatine Phosphokinase",
    98: "Echocardiogram",
    99: "Hemoglobin A1C",
    100: "Lactate",
    101: "Lipase",
    102: "Lumbar Puncture_CSF Analysis",
    103: "MRI",
    104: "Procalcitonin",
    105: "Pulse Oximetry_Arterial Saturation",
    106: "Stress Test (exercise or pharmacologic)",
    107: "Thyroid Stimulating Hormone (TSH)",
    108: "Toxicology Screen",
    109: "Troponin I_T",
    110: "Ultrasound Doppler (extremities)",
    111: "Ultrasound FAST exam",
    112: "Ultrasound Other",
    113: "Urinalysis",
    114: "Urine Pregnancy Test",
}


def find_next_id() -> int:
    """Find the next available statement ID by scanning raw and processed directories."""
    statement_ids = []

    # Scan raw statements
    raw_statements = glob.glob("../data/raw/train/statements/statement_*.txt")
    for filepath in raw_statements:
        filename = os.path.basename(filepath)
        statement_id = int(filename.split("_")[1].split(".")[0])
        statement_ids.append(statement_id)

    # Scan processed statements (all synthetic directories)
    for subdir in ["true", "false", "syntetic_true", "syntetic_false"]:
        processed_dir = f"../data/processed/{subdir}"
        if os.path.exists(processed_dir):
            processed_statements = glob.glob(f"{processed_dir}/statement_*.txt")
            for filepath in processed_statements:
                filename = os.path.basename(filepath)
                statement_id = int(filename.split("_")[1].split(".")[0])
                statement_ids.append(statement_id)

    return max(statement_ids) + 1 if statement_ids else 200


def load_topic_articles(topic_id: int) -> str:
    """Load all markdown articles for a given topic."""
    topic_name = TOPIC_MAPPING.get(topic_id)
    if not topic_name:
        raise ValueError(f"Unknown topic ID: {topic_id}")

    topic_dir = f"../data/topics/{topic_name}"
    if not os.path.exists(topic_dir):
        raise FileNotFoundError(f"Topic directory not found: {topic_dir}")

    articles_content = ""
    md_files = glob.glob(f"{topic_dir}/*.md")

    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
            articles_content += f"\n\n--- {os.path.basename(md_file)} ---\n{content}"

    return articles_content


def load_existing_statements() -> List[str]:
    """Load all existing statements from raw and processed directories for deduplication."""
    statements = []

    # Load raw statements
    raw_statements = glob.glob("../data/raw/train/statements/statement_*.txt")
    for filepath in raw_statements:
        with open(filepath, "r", encoding="utf-8") as f:
            statements.append(f.read().strip())

    # Load processed statements (all synthetic directories)
    for subdir in ["true", "false", "syntetic_true", "syntetic_false"]:
        processed_dir = f"../data/processed/{subdir}"
        if os.path.exists(processed_dir):
            processed_statements = glob.glob(f"{processed_dir}/statement_*.txt")
            for filepath in processed_statements:
                with open(filepath, "r", encoding="utf-8") as f:
                    statements.append(f.read().strip())

    return statements


def generate_statements_with_claude(
    topic_id: int, needed_true: int, needed_false: int
) -> Dict[str, List[str]]:
    """Generate statements using Claude API."""
    articles = load_topic_articles(topic_id)
    topic_name = TOPIC_MAPPING[topic_id]

    prompt = f"""You are a medical expert creating training data for an emergency healthcare AI system.

Based on these reference articles about "{topic_name}":

{articles}

Generate exactly {needed_true} TRUE statements and {needed_false} FALSE statements about this topic.

Requirements:
- Each statement must be a single sentence
- Maximum 40 words per statement
- TRUE statements must be factually accurate based on the articles
- FALSE statements must alter exactly one medical fact to make it incorrect
- Use plain English, medical terminology is fine
- Each statement should be unique and distinct
- Focus on key diagnostic criteria, treatments, symptoms, or procedures

Return your response as a JSON object with this exact format:
{{
  "true": [list of {needed_true} true statements],
  "false": [list of {needed_false} false statements]
}}

Ensure the JSON is valid and properly formatted."""

    try:
        response = client.messages.create(
            model=MODEL, max_tokens=2048, messages=[{"role": "user", "content": prompt}]
        )

        # Parse the JSON response
        response_text = response.content[0].text

        # Try to extract JSON from the response
        if "```json" in response_text:
            # Extract JSON from code block
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        else:
            # Assume the entire response is JSON
            json_str = response_text.strip()

        statements = json.loads(json_str)

        # Validate the structure
        if (
            not isinstance(statements, dict)
            or "true" not in statements
            or "false" not in statements
        ):
            raise ValueError("Invalid response structure")

        return statements

    except Exception as e:
        print(f"Error generating statements with Claude: {e}")
        raise


def deduplicate_statements(
    new_statements: List[str], existing_statements: List[str], threshold: float = 0.9
) -> List[str]:
    """Remove statements that are too similar to existing ones using sentence embeddings."""
    if not existing_statements:
        return new_statements

    # Embed all statements
    existing_embeddings = embedder.encode(existing_statements)
    new_embeddings = embedder.encode(new_statements)

    deduplicated = []
    for i, new_stmt in enumerate(new_statements):
        # Calculate similarity with all existing statements
        similarities = util.cos_sim(new_embeddings[i], existing_embeddings)
        max_similarity = float(similarities.max())

        if max_similarity < threshold:
            deduplicated.append(new_stmt)
        else:
            print(
                f"Skipping duplicate statement (similarity: {max_similarity:.3f}): {new_stmt[:50]}..."
            )

    return deduplicated


def save_statement_files(
    statement_id: int, statement_text: str, is_true: bool, topic_id: int
) -> None:
    """Save a single statement and its corresponding JSON answer file."""
    # Determine the directory
    subdir = "true" if is_true else "false"
    statement_dir = f"../data/processed/{subdir}"

    # Ensure directory exists
    os.makedirs(statement_dir, exist_ok=True)

    # File paths
    statement_file = f"{statement_dir}/statement_{statement_id:04d}.txt"
    answer_file = f"{statement_dir}/statement_{statement_id:04d}.json"

    # Write statement file (plain text, no extra newlines)
    with open(statement_file, "w", encoding="utf-8") as f:
        f.write(statement_text)

    # Write answer file (JSON with exact format matching)
    answer_data = {
        "statement_is_true": 1 if is_true else 0,
        "statement_topic": topic_id,
    }

    with open(answer_file, "w", encoding="utf-8") as f:
        json.dump(answer_data, f, indent=2)


def generate_topic_data(topic_id: int, needed_true: int, needed_false: int) -> None:
    """Main function to generate topic data."""
    print(
        f"Generating data for topic {topic_id} ({TOPIC_MAPPING.get(topic_id, 'Unknown')})"
    )
    print(f"Needed: {needed_true} true, {needed_false} false statements")

    # Load existing statements for deduplication
    existing_statements = load_existing_statements()
    print(f"Loaded {len(existing_statements)} existing statements for deduplication")

    # Generate new statements
    raw_statements = generate_statements_with_claude(
        topic_id, needed_true, needed_false
    )

    # Deduplicate
    deduplicated_true = deduplicate_statements(
        raw_statements["true"], existing_statements
    )
    deduplicated_false = deduplicate_statements(
        raw_statements["false"], existing_statements
    )

    print(
        f"After deduplication: {len(deduplicated_true)} true, {len(deduplicated_false)} false statements"
    )

    # Get starting ID
    start_id = find_next_id()
    current_id = start_id

    # Save true statements
    for statement in deduplicated_true:
        save_statement_files(current_id, statement, True, topic_id)
        current_id += 1

    # Save false statements
    for statement in deduplicated_false:
        save_statement_files(current_id, statement, False, topic_id)
        current_id += 1

    total_generated = len(deduplicated_true) + len(deduplicated_false)
    end_id = start_id + total_generated - 1

    print(
        f" Topic {topic_id} | wrote {total_generated} new files | id range {start_id:04d}-{end_id:04d}"
    )


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for Emergency Healthcare RAG"
    )
    parser.add_argument(
        "--topic-id", type=int, required=True, help="Topic ID for generation (0-114)"
    )
    parser.add_argument(
        "--needed-true",
        type=int,
        required=True,
        help="Number of true statements needed",
    )
    parser.add_argument(
        "--needed-false",
        type=int,
        required=True,
        help="Number of false statements needed",
    )

    args = parser.parse_args()

    # Validate topic ID
    if args.topic_id not in TOPIC_MAPPING:
        print(f"Error: Invalid topic ID {args.topic_id}. Valid range: 0-114")
        return 1

    # Validate needed counts
    if args.needed_true < 0 or args.needed_false < 0:
        print("Error: Needed counts must be non-negative")
        return 1

    if args.needed_true == 0 and args.needed_false == 0:
        print("Nothing to generate - both needed counts are 0")
        return 0

    try:
        generate_topic_data(args.topic_id, args.needed_true, args.needed_false)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
