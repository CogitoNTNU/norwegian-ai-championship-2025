import argparse
import json
import logging
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm

link_re = re.compile(r"\[([^\]]+?)\]\([^)]+?\)")

# Download NLTK punkt tokenizer if not already present
try:
    import nltk
    from nltk.tokenize import sent_tokenize

    # Try to use sent_tokenize, download punkt if needed
    try:
        sent_tokenize("Test.")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            nltk.download("punkt", quiet=True)
except ImportError:
    # Fallback to simple period splitting if NLTK not available
    def sent_tokenize(text):
        # Simple sentence splitting on periods followed by space/newline/end
        sentences = re.split(r"\.[\s\n]+|\.$", text)
        return [s.strip() + "." for s in sentences if s.strip()]


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

CWD = Path(__file__).resolve().parent
DATA_DIR = CWD.parent / "data"
TOPICS_FILE = DATA_DIR / "topics.json"
ARTICLES_DIR = DATA_DIR / "raw/topics"
KG_DIR = CWD / "kg"
CHUNKS_FILE = KG_DIR / "chunks.jsonl"

MIN_CHUNK_WORDS = 200
MAX_CHUNK_WORDS = 320
OVERLAP_SENTENCES = 2  # Number of sentences to overlap between chunks


def clean_text(raw):
    """Comprehensive text cleaning with residual boilerplate removal."""
    lines = raw.split("\n")
    cleaned_lines = []
    in_references = False

    for line in lines:
        # Check for References section - stop processing once we hit it
        if re.match(r"^[#*\s]*References\s*$", line.strip(), re.IGNORECASE):
            in_references = True
            break

        # Skip if we're in references section
        if in_references:
            break

        # Skip boilerplate lines
        if (
            line.strip().startswith("## source:")
            or line.strip().startswith("**Disclosure:")
            or line.strip() == "_" * len(line.strip())  # Skip separator lines
            or not line.strip()
        ):
            continue

        # Skip StatPearls promo lines
        promo = line.lower()
        if (
            promo.startswith("access free multiple choice")
            or "comment on this article" in promo
        ):
            continue

        cleaned_lines.append(line)

    # Join lines and apply text cleaning
    txt = "\n".join(cleaned_lines)
    txt = unicodedata.normalize("NFKC", txt)

    # Strip markdown links but keep anchor text
    txt = link_re.sub(r"\1", txt)

    # Angle-bracket URLs
    txt = re.sub(r"<https?://[^>]+>", "", txt)

    # Evidence tags
    txt = re.sub(r"\[Level\s*\d+\]", "", txt)

    # Inline numeric citations [12] [12][13]
    txt = re.sub(r"\[\d+\](\[\d+\])*", "", txt)

    # Leftover empty links
    txt = re.sub(r"\[\s*\]\(\s*\)", "", txt)

    # Escaped markdown artefacts
    txt = txt.replace("\\*", "").replace("\\\\", "")

    # Additional promo line removal from text blocks
    lines = txt.split("\n")
    final_lines = []
    for line in lines:
        promo = line.lower()
        if not (
            promo.startswith("access free multiple choice")
            or "comment on this article" in promo
        ):
            final_lines.append(line)
    txt = "\n".join(final_lines)

    # Collapse >1 blank line to single \n
    txt = re.sub(r"\n{2,}", "\n", txt).strip()

    return txt


def extract_heading_level(line):
    """Extract heading level and text from markdown heading."""
    match = re.match(r"^(#+)\s*(.*)$", line.strip())
    if match:
        return len(match.group(1)), match.group(2).strip()
    return None, None


def split_sentences_smart(text, max_words):
    """Split text into sentences, respecting word count limits."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())

        if current_words + sentence_words > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_words = 0

        current_chunk.append(sentence)
        current_words += sentence_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def get_overlap_sentences(text, max_tokens=40):
    """Get last sentences from text for overlap, capped at max_tokens."""
    sentences = sent_tokenize(text)
    if not sentences:
        return ""

    # Work backwards through sentences until we hit token limit
    overlap_text = ""
    for i in range(len(sentences) - 1, -1, -1):
        candidate = sentences[i] + (" " + overlap_text if overlap_text else "")
        # Rough token count (1 token ≈ 0.75 words)
        token_count = len(candidate.split()) * 0.75
        if token_count > max_tokens:
            break
        overlap_text = candidate

    return overlap_text


def create_chunks(overwrite=False):
    logging.info(f"Working directory: {CWD}")
    logging.info(f"Data directory: {DATA_DIR}")
    logging.info(f"Articles directory: {ARTICLES_DIR}")
    logging.info(f"Output file: {CHUNKS_FILE}")

    if CHUNKS_FILE.exists() and not overwrite:
        logging.info("Chunks file already exists. Use --overwrite to rebuild.")
        return

    KG_DIR.mkdir(exist_ok=True)

    with open(TOPICS_FILE, "r", encoding="utf-8") as f:
        topics_map = json.load(f)

    chunk_id_counter = 0
    chunks = []

    # Sort by topic_id (values) to maintain order
    sorted_topics = sorted(topics_map.items(), key=lambda x: x[1])

    for topic_name, topic_id in tqdm(sorted_topics, desc="Processing topics"):
        # Look for files using topic name directory structure
        topic_dir = ARTICLES_DIR / topic_name
        if not topic_dir.is_dir():
            logging.warning(f"Directory {topic_dir} does not exist, skipping...")
            continue

        # Process each .md file separately to track article titles
        md_files = list(topic_dir.glob("*.md"))
        if not md_files:
            logging.warning(f"No .md files found in {topic_dir}, skipping...")
            continue

        for md_file in md_files:
            # Extract article title from filename (remove .md extension)
            article_title = md_file.stem

            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Clean the content
            content = clean_text(content)

            # Split into lines for processing
            lines = content.split("\n")

            current_section = []
            overlap_buffer = ""
            position_counter = 0

            i = 0
            while i < len(lines):
                line = lines[i].strip()

                # Check if this is a heading
                heading_level, heading_text = extract_heading_level(line)

                if heading_level is not None:
                    # Process current section before starting new one
                    if current_section:
                        section_text = "\n".join(current_section).strip()
                        if section_text:
                            section_chunks = process_section(
                                section_text,
                                overlap_buffer,
                                topic_id,
                                topic_name,
                                article_title,
                                position_counter,
                                chunk_id_counter,
                            )
                            chunks.extend(section_chunks)
                            chunk_id_counter += len(section_chunks)
                            position_counter += len(section_chunks)

                            # Set overlap for next section from last chunk
                            if section_chunks:
                                overlap_buffer = get_overlap_sentences(
                                    section_chunks[-1]["text"]
                                )

                    # Start new section with heading
                    current_section = [heading_text] if heading_text else []
                else:
                    # Add non-heading line to current section
                    if line:  # Skip empty lines
                        current_section.append(line)

                i += 1

            # Process final section
            if current_section:
                section_text = "\n".join(current_section).strip()
                if section_text:
                    section_chunks = process_section(
                        section_text,
                        overlap_buffer,
                        topic_id,
                        topic_name,
                        article_title,
                        position_counter,
                        chunk_id_counter,
                    )
                    chunks.extend(section_chunks)
                    chunk_id_counter += len(section_chunks)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    logging.info(f"Created {len(chunks)} chunks.")


def process_section(
    section_text,
    overlap_buffer,
    topic_id,
    topic_name,
    article_title,
    position_start,
    chunk_id_start,
):
    """Process a section into chunks with overlap."""
    chunks = []

    # Skip very short sections (likely noise)
    base_word_count = len(section_text.split())
    if base_word_count < 10:  # Skip very short sections
        return chunks

    # Add overlap from previous chunk if exists
    if overlap_buffer:
        section_text = overlap_buffer + " " + section_text

    word_count = len(section_text.split())

    if word_count <= MAX_CHUNK_WORDS:
        # Section fits in one chunk - only create if meets minimum size
        if word_count >= MIN_CHUNK_WORDS:
            # Normalize text
            normalized_text = unicodedata.normalize("NFKC", section_text).strip()
            chunks.append(
                {
                    "chunk_id": chunk_id_start,
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "article_title": article_title,
                    "position": position_start,
                    "word_count": len(normalized_text.split()),
                    "text": normalized_text,
                }
            )
    else:
        # Need to split section into multiple chunks
        text_chunks = split_sentences_smart(section_text, MAX_CHUNK_WORDS)

        for i, chunk_text in enumerate(text_chunks):
            chunk_word_count = len(chunk_text.split())

            # Add overlap from previous chunk (except for first chunk which already has it)
            if i > 0 and chunks:
                prev_overlap = get_overlap_sentences(chunks[-1]["text"])
                chunk_text = prev_overlap + " " + chunk_text
                chunk_word_count = len(chunk_text.split())

            # Only add chunk if it meets minimum word count
            if chunk_word_count >= MIN_CHUNK_WORDS:
                # Normalize text
                normalized_text = unicodedata.normalize("NFKC", chunk_text).strip()
                chunks.append(
                    {
                        "chunk_id": chunk_id_start + i,
                        "topic_id": topic_id,
                        "topic_name": topic_name,
                        "article_title": article_title,
                        "position": position_start + i,
                        "word_count": len(normalized_text.split()),
                        "text": normalized_text,
                    }
                )

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Chunk StatPearls articles.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing chunks file.",
    )
    args = parser.parse_args()

    create_chunks(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
