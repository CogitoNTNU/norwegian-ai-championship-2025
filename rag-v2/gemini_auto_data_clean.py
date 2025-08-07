"""
Medical Text Cleaning Script using Gemini API
Extracts medically relevant facts from structured medical documents
"""

import os
import re
import logging
import time
from pathlib import Path
from typing import Optional
import google.generativeai as genai
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MedicalTextCleaner:
    def __init__(self, api_key: str):
        """Initialize the cleaner with Gemini API key"""
        genai.configure(api_key=api_key)

        # Initialize the model
        self.model = genai.GenerativeModel(
            "gemini-1.5-flash"
        )  # Using Flash for cost efficiency

        # Custom prompt for identifying text to delete (much more cost-effective)
        self.cleaning_prompt = """You are a medical text editor. Your task is to identify NON-MEDICAL sections that should be DELETED from the provided medical document.

IDENTIFY FOR DELETION (output these exact text sections):
- Author information and affiliations
- Publication metadata and dates  
- Navigation elements and links
- Continuing education objectives and learning goals
- Review questions and quiz sections
- Reference lists and citations
- Disclosure statements
- Copyright and source information
- Table of contents
- Administrative text and boilerplate
- URLs and web links
- PMC/PubMed reference numbers
- Date stamps and version information
- Pictures/Images and their heading and text

KEEP (do NOT mark for deletion):
- Clinical facts and medical statements
- Diagnostic criteria and disease classifications
- Treatment protocols and drug information
- Statistics, percentages, and medical measurements
- Pathophysiology explanations
- Risk factors and complications
- Medical terminology and definitions

OUTPUT FORMAT:
For each section to delete, provide the EXACT text as it appears in the document, one per line.
Use this format:

DELETE_START
[exact text to delete 1]
DELETE_END

DELETE_START  
[exact text to delete 2]
DELETE_END

Only output sections that should be deleted. Be precise with the text matching.

Document to analyze:

"""

    def clean_text(self, text: str) -> Optional[str]:
        """Clean text by having model identify sections to delete, then removing them"""
        try:
            # Pre-process: remove obvious non-medical sections
            text = self._preprocess_text(text)

            # Call Gemini API to identify sections to delete
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(
                        self.cleaning_prompt + text,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.1,  # Low temperature for precise identification
                            max_output_tokens=1024,  # Smaller since we're only identifying sections
                            top_p=0.8,
                        ),
                    )

                    if response.text:
                        # Parse the deletion instructions and apply them
                        cleaned_text = self._apply_deletions(text, response.text)

                        # Post-process: additional cleaning
                        cleaned_text = self._postprocess_text(cleaned_text)
                        return cleaned_text if cleaned_text else None
                    else:
                        logger.warning(
                            f"Empty response from Gemini on attempt {attempt + 1}"
                        )

                except Exception as e:
                    if "quota" in str(e).lower() or "rate" in str(e).lower():
                        logger.warning(
                            f"Rate limit hit, waiting before retry {attempt + 1}"
                        )
                        time.sleep(2**attempt)  # Exponential backoff
                    else:
                        logger.error(
                            f"Gemini API error on attempt {attempt + 1}: {str(e)}"
                        )

                    if attempt == max_retries - 1:
                        raise e

            return None

        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return None

    def _apply_deletions(self, original_text: str, deletion_instructions: str) -> str:
        """Apply the deletion instructions from the model to the original text"""
        # Parse deletion blocks
        deletion_pattern = r"DELETE_START\s*(.*?)\s*DELETE_END"
        deletions = re.findall(deletion_pattern, deletion_instructions, re.DOTALL)

        cleaned_text = original_text
        deletion_count = 0

        for deletion in deletions:
            deletion = deletion.strip()
            if (
                deletion and len(deletion) > 10
            ):  # Avoid deleting very short strings accidentally
                # Try exact match first
                if deletion in cleaned_text:
                    cleaned_text = cleaned_text.replace(deletion, "", 1)
                    deletion_count += 1
                else:
                    # Try with normalized whitespace
                    normalized_deletion = re.sub(r"\s+", " ", deletion.strip())
                    normalized_text = re.sub(r"\s+", " ", cleaned_text)

                    if normalized_deletion in normalized_text:
                        # Find the original text span and replace it
                        pattern = re.escape(normalized_deletion).replace(r"\ ", r"\s+")
                        cleaned_text = re.sub(
                            pattern, "", cleaned_text, count=1, flags=re.MULTILINE
                        )
                        deletion_count += 1

        logger.info(f"Applied {deletion_count} deletions")

        # Clean up extra whitespace after deletions
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)

        return cleaned_text.strip()

    def _preprocess_text(self, text: str) -> str:
        """Pre-process text to remove obvious non-medical content"""

        # First, remove everything after "## References" (including the References section)
        references_pattern = r"## References.*$"
        text = re.sub(references_pattern, "", text, flags=re.DOTALL | re.MULTILINE)

        # Also handle variations of references sections
        references_variations = [
            r"## Reference.*$",
            r"# References.*$",
            r"# Reference.*$",
            r"References\n.*$",
            r"REFERENCES\n.*$",
        ]

        for pattern in references_variations:
            text = re.sub(pattern, "", text, flags=re.DOTALL | re.MULTILINE)

        # Remove other common metadata patterns
        patterns_to_remove = [
            r"Author Information and Affiliations.*?(?=##|\n\n)",
            r"Disclosure:.*?(?=\n\n)",
            r"Last Update:.*?(?=\n)",
            r"Continuing Education Activity.*?(?=##)",
            r"Review Questions.*?$",
            r"Access free multiple choice questions.*?(?=\n)",
            r"Click here for.*?(?=\n)",
            r"Comment on this article.*?(?=\n)",
            r"source: https://.*?scraped_date:.*?\n",
            r"_{5,}",  # Long underscores
            r"\[PMC.*?\]",  # PMC references
            r"\[PubMed:.*?\]",  # PubMed references
        ]

        for pattern in patterns_to_remove:
            text = re.sub(pattern, "", text, flags=re.DOTALL | re.MULTILINE)

        # Clean up extra whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        content = text

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
            r"#{1,6}\s*Affiliations?\s*\n.*?(?=\n#{1,6}\s|\Z)",
            "",
            content,
            flags=re.DOTALL,
        )

        # Remove disclosure statements
        content = re.sub(
            r"\*\*Disclosure:\*\*.*?(?=\n|\Z)", "", content, flags=re.DOTALL
        )

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
        content = re.sub(
            r"\[!\[.*?\]\(.*?\)\]\(.*?\)", "", content
        )  # Complex figure links
        content = re.sub(r"!\[.*?\]\(.*?\)", "", content)  # Simple image links
        content = re.sub(
            r"#{1,6}\s*\[?Figure\]?.*?(?=\n#{1,6}\s|\n\n|\Z)",
            "",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        content = re.sub(
            r"\(see\s*\*\*Image\*\*.*?\)", "", content, flags=re.IGNORECASE
        )
        content = re.sub(
            r"see\s*\*\*Image\.\*\*[^)]*", "", content, flags=re.IGNORECASE
        )

        # Remove URLs in brackets
        content = re.sub(
            r"\[([^\]]+)\]\([^\)]+\)", r"\1", content
        )  # Keep text, remove URL

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

    def _postprocess_text(self, text: str) -> str:
        """Post-process the cleaned text"""
        if not text:
            return ""

        # Remove any remaining metadata that might have slipped through
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip lines that are clearly metadata/formatting
            skip_patterns = [
                r"^\d+\.$",  # Just numbers
                r"^[A-Z\s]+:$",  # All caps headers
                r"^Source:",
                r"^Author:",
                r"^DOI:",
                r"^Published:",
                r"^\[[^\]]+\]$",  # References in brackets
            ]

            if any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def process_directory(
        self, input_dir: str, output_dir: str, delay_seconds: float = 1.0
    ):
        """Process all markdown files in directory structure"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.exists():
            logger.error(f"Input directory {input_dir} does not exist")
            return

        # Find all markdown files
        md_files = list(input_path.rglob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files to process")

        processed_count = 0
        skipped_count = 0

        for file_path in tqdm(md_files, desc="Processing files"):
            try:
                # Read the file
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Clean the content
                cleaned_content = self.clean_text(content)

                if cleaned_content:
                    # Create output directory structure
                    relative_path = file_path.relative_to(input_path)
                    output_file_path = output_path / relative_path
                    output_file_path.parent.mkdir(parents=True, exist_ok=True)

                    # Write cleaned content
                    logger.info(cleaned_content)
                    with open(output_file_path, "w", encoding="utf-8") as f:
                        f.write(cleaned_content)

                    processed_count += 1
                    logger.info(f"Processed: {relative_path}")
                else:
                    skipped_count += 1
                    logger.warning(
                        f"Skipped (insufficient content): {file_path.relative_to(input_path)}"
                    )

                # Rate limiting - add delay between requests
                if delay_seconds > 0:
                    time.sleep(delay_seconds)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                skipped_count += 1

        logger.info(
            f"Processing complete. Processed: {processed_count}, Skipped: {skipped_count}"
        )


def main():
    """Main function"""
    # Configuration
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        print("Please set your GEMINI_API_KEY environment variable")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return

    INPUT_DIR = "data/topics"
    OUTPUT_DIR = "data/topics_cleaned"

    # Initialize cleaner
    cleaner = MedicalTextCleaner(API_KEY)

    # Process directory with rate limiting
    cleaner.process_directory(INPUT_DIR, OUTPUT_DIR, delay_seconds=1.0)


if __name__ == "__main__":
    main()
