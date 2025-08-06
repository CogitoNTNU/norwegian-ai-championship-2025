#!/usr/bin/env python3
import argparse
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import List

try:
    from tqdm import tqdm  # pretty progress-bar
except ModuleNotFoundError:
    # fallback: behave like plain iterable
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


# ──────────────────────────────────────────────────────────────────────────────
# basic text helpers
link_re = re.compile(r"\[([^\]]+?)\]\([^)]+?\)")

try:
    import nltk
    from nltk.tokenize import sent_tokenize

    try:
        sent_tokenize("Test.")
    except LookupError:
        nltk.download("punkt", quiet=True)
except ImportError:

    def sent_tokenize(text: str) -> List[str]:
        parts = re.split(r"\.[\s\n]+|\.$", text)
        return [p.strip() + "." for p in parts if p.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# paths & constants
CWD = Path(__file__).resolve().parent
DATA_DIR = CWD.parent / "data"
TOPICS_FILE = DATA_DIR / "topics.json"
ARTICLES_DIR = DATA_DIR / "raw/topics"

KG_DIR = CWD / "kg"
KG_DIR.mkdir(exist_ok=True)
CHUNKS_FILE = KG_DIR / "chunks2.jsonl"

WINDOW_TOKENS = 120  # ≈ 90 words
STRIDE_TOKENS = 60
MIN_WINDOW_WORDS = 40

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# ──────────────────────────────────────────────────────────────────────────────
# cleaning & utility functions
def clean_text(raw: str) -> str:
    """Remove StatPearls boilerplate, markdown artefacts, refs, links."""
    lines, cleaned_lines = raw.split("\n"), []
    for line in lines:
        if re.match(r"^[#*\s]*References\s*$", line.strip(), re.I):
            break
        if (
            line.strip().startswith(("## source:", "**Disclosure:"))
            or line.strip() == "_" * len(line.strip())
            or not line.strip()
        ):
            continue
        promo = line.lower()
        if (
            promo.startswith("access free multiple choice")
            or "comment on this article" in promo
        ):
            continue
        cleaned_lines.append(line)

    txt = "\n".join(cleaned_lines)
    txt = unicodedata.normalize("NFKC", txt)
    txt = link_re.sub(r"\1", txt)
    txt = re.sub(r"<https?://[^>]+>", "", txt)  # <url>
    txt = re.sub(r"\[Level\s*\d+\]", "", txt)  # evidence tags
    txt = re.sub(r"\[\d+\](\[\d+\])*", "", txt)  # [12] [12][13]
    txt = re.sub(r"\[\s*\]\(\s*\)", "", txt)
    txt = txt.replace("\\*", "").replace("\\\\", "")
    txt = re.sub(r"\n{2,}", "\n", txt).strip()
    return txt


def extract_heading(line: str):
    m = re.match(r"^(#+)\s*(.*)$", line.strip())
    return (len(m.group(1)), m.group(2).strip()) if m else (None, None)


def sliding_windows(words, size, stride):
    for start in range(0, len(words), stride):
        window = words[start : start + size]
        if window:
            yield window


# ──────────────────────────────────────────────────────────────────────────────
def create_chunks(force: bool = False):
    if CHUNKS_FILE.exists() and not force:
        logging.info(
            f"{CHUNKS_FILE.name} already exists – skip (use --force to rebuild)"
        )
        return

    with open(TOPICS_FILE, encoding="utf-8") as f:
        topics_map = json.load(f)

    chunks, chunk_id = [], 0
    sorted_topics = sorted(topics_map.items(), key=lambda x: x[1])

    for topic_name, topic_id in tqdm(sorted_topics, desc="Scanning topics"):
        topic_dir = ARTICLES_DIR / topic_name
        if not topic_dir.is_dir():
            logging.warning(f"Dir missing: {topic_dir}")
            continue

        for md_file in topic_dir.glob("*.md"):
            article_title = md_file.stem
            raw = md_file.read_text(encoding="utf-8")
            content = clean_text(raw)

            lines = content.split("\n")
            section_lines: List[str] = []
            position_counter = 0

            for line in lines + ["# END"]:  # sentinel flush
                h_lvl, h_text = extract_heading(line)
                if h_lvl is not None or line == "# END":
                    if section_lines:
                        section_text = "\n".join(section_lines).strip()
                        words = section_text.split()
                        for win_words in sliding_windows(
                            words, WINDOW_TOKENS, STRIDE_TOKENS
                        ):
                            if len(win_words) < MIN_WINDOW_WORDS:
                                continue
                            text = unicodedata.normalize("NFKC", " ".join(win_words))
                            chunks.append(
                                {
                                    "chunk_id": chunk_id,
                                    "topic_id": topic_id,
                                    "topic_name": topic_name,
                                    "article_title": article_title,
                                    "position": position_counter,
                                    "word_count": len(text.split()),
                                    "text": text,
                                }
                            )
                            chunk_id += 1
                            position_counter += 1
                    section_lines = [h_text] if h_lvl else []
                else:
                    if line.strip():
                        section_lines.append(line.strip())

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch) + "\n")

    avg_words = sum(c["word_count"] for c in chunks) / len(chunks)
    logging.info(
        f" Wrote {len(chunks):,} chunks → {CHUNKS_FILE} (avg {avg_words:.1f} words)"
    )


# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser("Sliding-window chunker for StatPearls.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if chunks2.jsonl already exists.",
    )
    args = parser.parse_args()
    create_chunks(force=args.force)


if __name__ == "__main__":
    main()
