import json
from pathlib import Path

# Get the path to chunks.jsonl relative to the test file
TEST_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = TEST_DIR / "kg" / "chunks.jsonl"


def test_chunk_file_exists():
    assert CHUNKS_FILE.exists()


def test_chunk_counts():
    n = sum(1 for _ in open(CHUNKS_FILE, "r"))
    assert n > 500  # sanity: expect >500 chunks total after cleaning
    assert n < 5000  # sanity: expect <5000 chunks total


def test_chunk_format():
    with open(CHUNKS_FILE, "r") as f:
        first_line = f.readline()
        chunk = json.loads(first_line)

        # Check required fields
        required_fields = [
            "chunk_id",
            "topic_id",
            "topic_name",
            "article_title",
            "position",
            "word_count",
            "text",
        ]
        for field in required_fields:
            assert field in chunk

        # Check data types
        assert isinstance(chunk["chunk_id"], int)
        assert isinstance(chunk["topic_id"], int)
        assert isinstance(chunk["topic_name"], str)
        assert isinstance(chunk["article_title"], str)
        assert isinstance(chunk["position"], int)
        assert isinstance(chunk["word_count"], int)
        assert isinstance(chunk["text"], str)


def test_word_count_range():
    with open(CHUNKS_FILE, "r") as f:
        chunks_checked = 0
        for line in f:
            chunk = json.loads(line)
            word_count = chunk["word_count"]
            actual_word_count = len(chunk["text"].split())

            # Verify word count is accurate
            assert word_count == actual_word_count

            chunks_checked += 1
            if chunks_checked >= 100:  # Sample check first 100 chunks
                break


def test_article_title_field():
    """Test that article_title field is properly populated and non-empty."""
    with open(CHUNKS_FILE, "r") as f:
        chunks_checked = 0
        article_titles_seen = set()
        for line in f:
            chunk = json.loads(line)
            article_title = chunk["article_title"]

            # Article title should be non-empty string
            assert isinstance(article_title, str)
            assert len(article_title.strip()) > 0

            # Should not contain file extension
            assert not article_title.endswith(".md")

            article_titles_seen.add(article_title)

            chunks_checked += 1
            if chunks_checked >= 200:  # Sample check first 200 chunks
                break

        # Should have seen multiple different article titles
        assert len(article_titles_seen) > 1
