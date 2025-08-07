from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

from embeddings import get_embeddings_func
from get_config import config

import os
import argparse
import shutil
import re


def preprocess_document_content(content):
    """
    Preprocess document content to remove unnecessary information
    while preserving factual medical content.
    """

    # Remove author information sections
    content = re.sub(
        r"#{1,6}\s*Author Information and Affiliations.*?(?=\n#{1,6}\s|\Z)",
        "",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    content = re.sub(
        r"#{1,6}\s*Authors?\s*\n.*?(?=\n#{1,6}\s|\Z)", "", content, flags=re.DOTALL
    )
    content = re.sub(
        r"#{1,6}\s*Affiliations?\s*\n.*?(?=\n#{1,6}\s|\Z)", "", content, flags=re.DOTALL
    )

    # Remove disclosure statements
    content = re.sub(r"\*\*Disclosure:\*\*.*?(?=\n|\Z)", "", content, flags=re.DOTALL)

    # Remove references section and everything after it
    content = re.sub(
        r"#{1,6}\s*References\s*\n.*", "", content, flags=re.DOTALL | re.IGNORECASE
    )

    # Remove review questions sections
    content = re.sub(
        r"#{1,6}\s*Review Questions.*?(?=\n#{1,6}\s|\Z)",
        "",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove figure references and descriptions
    content = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)", "", content)  # Complex figure links
    content = re.sub(r"!\[.*?\]\(.*?\)", "", content)  # Simple image links
    content = re.sub(
        r"#{1,6}\s*\[?Figure\]?.*?(?=\n#{1,6}\s|\n\n|\Z)",
        "",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    content = re.sub(r"\(see\s*\*\*Image\*\*.*?\)", "", content, flags=re.IGNORECASE)
    content = re.sub(r"see\s*\*\*Image\.\*\*[^)]*", "", content, flags=re.IGNORECASE)

    # Remove URLs in brackets
    content = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", content)  # Keep text, remove URL

    # Remove PMC and PubMed references
    content = re.sub(r"\\\?\[PMC.*?\\\?\]", "", content)
    content = re.sub(r"\\\?\[PubMed:?\s*\d+\\\?\]", "", content)
    content = re.sub(r"\[PMC free article:.*?\]", "", content)
    content = re.sub(r"\[PubMed:.*?\]", "", content)

    # Remove access/click instructions
    content = re.sub(r"\[Access free.*?\]", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\[Click here.*?\]", "", content, flags=re.IGNORECASE)
    content = re.sub(
        r"\[Comment on this article.*?\]", "", content, flags=re.IGNORECASE
    )

    # Remove metadata like scraped date and source URL
    content = re.sub(r"_{3,}", "", content)  # Remove separator lines
    content = re.sub(r"##\s*source:.*?(?=\n)", "", content, flags=re.IGNORECASE)
    content = re.sub(r"scraped_date:.*?(?=\n)", "", content, flags=re.IGNORECASE)

    # Remove Last Update lines
    content = re.sub(r"Last Update:.*?(?=\n)", "", content, flags=re.IGNORECASE)

    # Remove numbered citations in square brackets
    content = re.sub(r"\[\d+\](?:\[\d+\])*", "", content)

    # Remove "Contributed by" statements
    content = re.sub(r"Contributed by.*?(?=\n|\Z)", "", content)

    # Remove excessive whitespace
    content = re.sub(r"\n{3,}", "\n\n", content)  # Max 2 consecutive newlines
    content = re.sub(r" {2,}", " ", content)  # Multiple spaces to single space

    # Remove any remaining empty headers
    content = re.sub(r"#{1,6}\s*\n", "", content)

    # Strip leading/trailing whitespace
    content = content.strip()

    return content


def expand_medical_abbreviations(content):
    """
    Expand common medical abbreviations to improve searchability.
    Adds the expansion in parentheses after the abbreviation.
    """
    # Define common medical abbreviations and their expansions
    abbreviations = {
        "CHD": "coronary heart disease",
        "MI": "myocardial infarction heart attack",
        "CPR": "cardiopulmonary resuscitation",
        "ACS": "acute coronary syndrome",
        "PE": "pulmonary embolism",
        "HTN": "hypertension high blood pressure",
        "DM": "diabetes mellitus",
        "COPD": "chronic obstructive pulmonary disease",
        "DVT": "deep vein thrombosis blood clot",
        "ICU": "intensive care unit",
        "ED": "emergency department",
        "EKG": "electrocardiogram ECG",
        "ECG": "electrocardiogram EKG",
        "BP": "blood pressure",
        "HR": "heart rate",
        "STEMI": "ST elevation myocardial infarction",
        "NSTEMI": "non ST elevation myocardial infarction",
        "CAD": "coronary artery disease",
        "CHF": "congestive heart failure",
        "HF": "heart failure",
        "AFib": "atrial fibrillation",
        "AF": "atrial fibrillation",
        "VT": "ventricular tachycardia",
        "VF": "ventricular fibrillation",
        "PCI": "percutaneous coronary intervention",
        "CABG": "coronary artery bypass graft surgery",
        "TIA": "transient ischemic attack mini stroke",
        "CVA": "cerebrovascular accident stroke",
        "CKD": "chronic kidney disease",
        "ESRD": "end stage renal disease",
        "GFR": "glomerular filtration rate",
        "BUN": "blood urea nitrogen",
        "LDL": "low density lipoprotein bad cholesterol",
        "HDL": "high density lipoprotein good cholesterol",
        "TG": "triglycerides",
        "A1C": "hemoglobin A1C glycated hemoglobin",
        "HbA1c": "hemoglobin A1C glycated hemoglobin",
        "CBC": "complete blood count",
        "WBC": "white blood cell count",
        "RBC": "red blood cell count",
        "Hgb": "hemoglobin",
        "Hct": "hematocrit",
        "BNP": "brain natriuretic peptide",
        "TNF": "tumor necrosis factor",
        "CT": "computed tomography CAT scan",
        "MRI": "magnetic resonance imaging",
        "US": "ultrasound",
        "CXR": "chest x-ray",
        "ABG": "arterial blood gas",
        "ARDS": "acute respiratory distress syndrome",
        "OSA": "obstructive sleep apnea",
        "URI": "upper respiratory infection",
        "UTI": "urinary tract infection",
        "GI": "gastrointestinal",
        "GERD": "gastroesophageal reflux disease",
        "IBD": "inflammatory bowel disease",
        "IBS": "irritable bowel syndrome",
        "NASH": "non alcoholic steatohepatitis",
        "NAFLD": "non alcoholic fatty liver disease",
        "ALT": "alanine aminotransferase",
        "AST": "aspartate aminotransferase",
        "PT": "prothrombin time",
        "PTT": "partial thromboplastin time",
        "INR": "international normalized ratio",
        "PTE": "pulmonary thromboembolism",
        "AAA": "abdominal aortic aneurysm",
        "PAD": "peripheral artery disease",
        "PVD": "peripheral vascular disease",
        "SBP": "systolic blood pressure",
        "DBP": "diastolic blood pressure",
        "MAP": "mean arterial pressure",
        "CVP": "central venous pressure",
        "PCWP": "pulmonary capillary wedge pressure",
        "CO": "cardiac output",
        "CI": "cardiac index",
        "EF": "ejection fraction",
        "LV": "left ventricle left ventricular",
        "RV": "right ventricle right ventricular",
        "LA": "left atrium left atrial",
        "RA": "right atrium right atrial",
        "AV": "atrioventricular",
        "SA": "sinoatrial",
        "LAD": "left anterior descending artery",
        "RCA": "right coronary artery",
        "LCX": "left circumflex artery",
        "PDA": "posterior descending artery",
        "LMCA": "left main coronary artery",
        "SVT": "supraventricular tachycardia",
        "NSR": "normal sinus rhythm",
        "PVC": "premature ventricular contraction",
        "PAC": "premature atrial contraction",
        "LBBB": "left bundle branch block",
        "RBBB": "right bundle branch block",
        "IVCD": "intraventricular conduction delay",
        "QTc": "corrected QT interval",
        "ROSC": "return of spontaneous circulation",
        "DNR": "do not resuscitate",
        "DNI": "do not intubate",
        "ACLS": "advanced cardiac life support",
        "BLS": "basic life support",
        "AED": "automated external defibrillator",
        "ICD": "implantable cardioverter defibrillator",
        "CRT": "cardiac resynchronization therapy",
        "VAD": "ventricular assist device",
        "ECMO": "extracorporeal membrane oxygenation",
        "IABP": "intra aortic balloon pump",
    }

    # Track which abbreviations have already been expanded to avoid duplicates
    expanded = set()

    for abbr, expansion in abbreviations.items():
        # Use word boundaries to match whole words only
        # Check if abbreviation exists and hasn't been expanded yet
        pattern = r"\b" + re.escape(abbr) + r"\b(?![^(]*\))"

        # Check if this abbreviation appears in the content
        if re.search(pattern, content) and abbr not in expanded:
            # Add expansion after first occurrence
            # Replace only the first occurrence with abbreviation + expansion
            replacement = f"{abbr} ({expansion})"
            content = re.sub(pattern, replacement, content, count=1)
            expanded.add(abbr)

            # For subsequent occurrences, leave the abbreviation as is
            # This avoids cluttering the text with repeated expansions

    return content


def load_documents():
    """Load all markdown documents from all topic folders in the data directory."""
    base_path = Path("data/topics")
    documents = []

    # Iterate through all subdirectories (topics) in the data folder
    for topic_folder in base_path.iterdir():
        if topic_folder.is_dir():
            topic_name = topic_folder.name
            print(f"\n📁 Processing topic: {topic_name}")

            # Get all .md files in the topic folder
            md_files = topic_folder.glob("*.md")

            for file_path in md_files:
                print(f"  Loading: {file_path.name}")
                try:
                    loader = UnstructuredMarkdownLoader(str(file_path))
                    docs = loader.load()

                    # Extract filename without extension for topic keywords
                    file_topic = (
                        file_path.stem
                    )  # e.g., "ecg_t_wave" from "ecg_t_wave.md"

                    # Create human-readable topic keywords from filename
                    # Convert underscores to spaces and add to searchable content
                    topic_keywords = file_topic.replace("_", " ").replace("-", " ")

                    # Preprocess each document's content
                    for doc in docs:
                        # Apply preprocessing to clean the content
                        cleaned_content = preprocess_document_content(doc.page_content)

                        # Expand medical abbreviations for better searchability
                        cleaned_content = expand_medical_abbreviations(cleaned_content)

                        # Prepend topic information to the content for better searchability
                        # This ensures topic-related queries match the document
                        enhanced_content = f"Topic: {topic_name} - {topic_keywords}\n\n{cleaned_content}"
                        doc.page_content = enhanced_content

                        # Add comprehensive metadata
                        doc.metadata["topic"] = topic_name
                        doc.metadata["file_topic"] = file_topic
                        doc.metadata["topic_keywords"] = topic_keywords
                        doc.metadata["source_file"] = file_path.name

                        # Only add document if it has substantial content after cleaning
                        if len(cleaned_content.strip()) < 100:
                            print(
                                f"  ⚠️ Skipping {file_path.name} - insufficient content after preprocessing"
                            )
                            continue

                    # Filter out empty documents
                    docs = [doc for doc in docs if len(doc.page_content.strip()) >= 100]
                    documents.extend(docs)

                except Exception as e:
                    print(f"  ❌ Error loading {file_path}: {e}")
                    continue

            # Count files processed for this topic
            topic_file_count = len(list(topic_folder.glob("*.md")))
            print(f"  ✓ Processed {topic_file_count} files in {topic_name}")

    print(f"\n📊 Total documents loaded: {len(documents)}")
    return documents


def split_documents(documents):
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],  # Prioritize paragraph and sentence boundaries
    )
    return splitter.split_documents(documents)


def calculate_chunk_ids(chunks: list[Document]):
    """Calculate unique IDs for each chunk including topic information."""
    for i, chunk in enumerate(chunks):
        # Create a more descriptive ID that includes the topic
        topic = chunk.metadata.get("topic", "unknown")
        file_topic = chunk.metadata.get("file_topic", "unknown")

        # Create ID format: topic:file_topic:chunk_index
        chunk_index = f"{topic}:{file_topic}:{i}"
        chunk.metadata["id"] = chunk_index

    return chunks


def add_to_chroma(chunks):
    """Add chunks to Chroma database."""
    db = Chroma(
        persist_directory=config["chroma_path"],
        embedding_function=get_embeddings_func(),
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Group new chunks by topic for reporting
    topic_counts = {}
    new_chunks = []

    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
            topic = chunk.metadata.get("topic", "unknown")
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

    if len(new_chunks):
        print(f"\n👉 Adding {len(new_chunks)} new documents in batches...")

        # Process in batches of 1000 to avoid ChromaDB limits
        batch_size = 1000
        for i in range(0, len(new_chunks), batch_size):
            batch_chunks = new_chunks[i : i + batch_size]
            batch_ids = [chunk.metadata["id"] for chunk in batch_chunks]
            db.add_documents(batch_chunks, ids=batch_ids)

        print("✅ Documents added to ChromaDB")
    else:
        print("✅ No new documents to add")


def list_topics():
    """List all available topics in the data directory."""
    base_path = Path("data/topics")
    topics = [d.name for d in base_path.iterdir() if d.is_dir()]

    if topics:
        print("\n📚 Available topics:")
        for topic in sorted(topics):
            file_count = len(list((base_path / topic).glob("*.md")))
            print(f"  - {topic} ({file_count} files)")
    else:
        print("⚠️ No topic folders found in data directory")

    return topics


def clear_database():
    if os.path.exists(config["chroma_path"]):
        shutil.rmtree(config["chroma_path"])


def preview_preprocessing(sample_file=None):
    """
    Preview the preprocessing on a sample file to verify cleaning.
    Useful for debugging and validation.
    """
    if sample_file:
        with open(sample_file, "r", encoding="utf-8") as f:
            original = f.read()

        # Extract topic information from file path
        file_path = Path(sample_file)
        file_topic = file_path.stem
        topic_keywords = file_topic.replace("_", " ").replace("-", " ")

        # Get topic folder name if available
        topic_name = (
            file_path.parent.name if file_path.parent.name != "data" else "general"
        )

        # Apply all preprocessing steps
        processed = preprocess_document_content(original)
        expanded = expand_medical_abbreviations(processed)
        enhanced = f"Topic: {topic_name} - {topic_keywords}\n\n{expanded}"

        print("=" * 50)
        print("ORIGINAL (first 500 chars):")
        print("=" * 50)
        print(original[:500])
        print("\n" + "=" * 50)
        print("PROCESSED WITH TOPIC AND EXPANSIONS (first 700 chars):")
        print("=" * 50)
        print(enhanced[:700])
        print("\n" + "=" * 50)
        print(f"Original length: {len(original)} chars")
        print(f"After cleaning: {len(processed)} chars")
        print(f"After abbreviation expansion: {len(expanded)} chars")
        print(f"Final with topic: {len(enhanced)} chars")
        print(
            f"Reduction from original: {100 * (1 - len(processed) / len(original)):.1f}%"
        )
        print("\nExtracted metadata:")
        print(f"  Topic: {topic_name}")
        print(f"  File topic: {file_topic}")
        print(f"  Topic keywords: {topic_keywords}")

        # Show some examples of expanded abbreviations
        print("\nSample abbreviation expansions:")
        sample_abbrs = ["ECG", "MI", "CHD", "STEMI", "BP"]
        for abbr in sample_abbrs:
            if abbr in expanded:
                # Find context around the abbreviation
                idx = expanded.find(abbr)
                if idx != -1:
                    start = max(0, idx - 20)
                    end = min(len(expanded), idx + 50)
                    context = expanded[start:end]
                    print(f"  {abbr}: ...{context}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument(
        "--preview", type=str, help="Preview preprocessing on a sample file."
    )
    args = parser.parse_args()

    if args.preview:
        print("🔍 Previewing preprocessing...")
        preview_preprocessing(args.preview)
        return

    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    """Main function to orchestrate document loading and storage."""
    print("🚀 Starting document processing with preprocessing...\n")

    # List available topics
    topics = list_topics()

    if not topics:
        print("⚠️ No topics found to process")
        return

    # Load and process documents
    documents = load_documents()

    if documents:
        chunks = split_documents(documents)
        print(f"\n📄 Total chunks created: {len(chunks)}")
        add_to_chroma(chunks)
    else:
        print("⚠️ No documents found to process")


if __name__ == "__main__":
    main()
