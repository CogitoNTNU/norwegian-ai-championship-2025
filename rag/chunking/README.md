# StatPearls Article Chunking

This script processes StatPearls reference articles and creates well-structured text chunks for RAG (Retrieval-Augmented Generation) systems.

## Directory Structure

The script expects the following directory structure:

```
project-root/
├── rag/
│   ├── data/
│   │   ├── topics.json              # Topic mapping file
│   │   └── raw/
│   │       └── topics/              # Article directories
│   │           ├── Topic Name 1/    # Directory named after topic
│   │           │   ├── article1.md
│   │           │   └── article2.md
│   │           └── Topic Name 2/
│   │               └── article.md
│   └── chunking/                    # Working directory
│       ├── chunk_topics.py          # Main chunking script
│       ├── test_chunking.py         # Unit tests
│       ├── README.md                # This file
│       └── kg/
│           └── chunks.jsonl         # Output file (generated)
```

## Usage

From the `rag/chunking` directory:

```bash
# Create chunks (won't overwrite existing file)
python chunk_topics.py

# Rebuild chunks (overwrite existing file)
python chunk_topics.py --overwrite
```

## Output Format

Each chunk is saved as a JSON object in `kg/chunks.jsonl`:

```json
{
  "chunk_id": 0,
  "topic_id": 42,
  "topic_name": "Heart Failure",
  "article_title": "Acute Heart Failure",
  "position": 0,
  "word_count": 287,
  "text": "The actual chunk text content..."
}
```

## Features

- **Sentence-Complete Chunks**: Ensures chunks never break mid-sentence using NLTK tokenization
- **Section-Focused**: Uses markdown headings as hard boundaries for better topical coherence
- **Smart Overlapping**: Copies last 1-2 sentences (~≤40 tokens) from each chunk to the next for context continuity
- **Comprehensive Text Cleaning**:
  - **Reference Section Removal**: Automatically detects and stops processing at "References" headings
  - **Citation Stripping**: Removes inline citations like [12], [3][9], [14][15]
  - **Markdown Cleanup**: Removes escape characters (\*, \\) and converts links [text](url) → text
  - **Boilerplate Filtering**: Removes source lines, disclosure statements, and separator lines
  - **Unicode Normalization**: Applies NFKC normalization for consistent text encoding
  - **Whitespace Normalization**: Collapses multiple blank lines and trims whitespace
- **Optimal Sizing**: Targets 200-320 words per chunk with strict minimum enforcement (0% under-sized chunks)
- **Rich Metadata**: Includes topic information, article titles, position, and accurate word counts
- **Order Preservation**: Maintains topic and chunk ordering
- **Robust Processing**: Handles missing files/directories gracefully
- **Universal Paths**: Uses relative paths, works on any machine

## Testing

Run the unit tests from the `rag/chunking` directory:

```bash
python -m pytest test_chunking.py -v
```

## Dependencies

- nltk (for sentence tokenization)
- tqdm (for progress bars)
- Standard library modules: json, pathlib, argparse, logging, re

The script automatically downloads NLTK's punkt tokenizer on first run.

## Performance Metrics

After implementing comprehensive text cleaning:

- **Total chunks**: 1,908 from 206 unique articles across 115 topics
- **Article tracking**: Each chunk includes both topic name and specific article title for precise source attribution
- **Size distribution**: Targets 200-320 words per chunk with strict minimum enforcement (0% under-sized chunks)
- **Noise elimination**: 100% citation removal, 100% reference section removal, 100% escape character removal
- **Processing efficiency**: Maintains high content quality while ensuring sentence-complete, section-focused chunks
