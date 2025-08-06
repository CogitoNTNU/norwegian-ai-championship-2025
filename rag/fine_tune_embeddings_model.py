# creating data formatation for 
"""
Data formation for fine-tuning embedding models on medical statements.
Creates positive and negative pairs for contrastive learning.
"""

import os
import json
import random
import re
import unicodedata
from typing import List, Dict, Any, Tuple
import glob
from pathlib import Path

# Advanced chunking imports
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

# Advanced text processing functions
link_re = re.compile(r"\[([^\]]+?)\]\([^)]+?\)")

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


# Simple replacements for PyTorch/sentence-transformers classes
class Dataset:
    """Simple Dataset base class to replace torch.utils.data.Dataset"""
    pass

class InputExample:
    """Simple InputExample class to replace sentence_transformers.InputExample"""
    def __init__(self, texts: List[str], label: float):
        self.texts = texts
        self.label = label


class MedicalTripletDataset(Dataset):
    """Dataset for medical statement-document triplets."""
    
    def __init__(
        self,
        statements_dir: str,
        answers_dir: str,
        topics_json_path: str,
        topics_documents_dir: str,
        negative_ratio: int = 4,  # Increased for better hard negatives
        min_chunk_words: int = 50,   # Smaller chunks for embedding models
        max_chunk_words: int = 150,  # Max ~200 tokens for BERT models
        overlap_sentences: int = 1,  # Reduced overlap
        max_chunks_per_topic: int = 20,  # More chunks per topic
        max_tokens_per_example: int = 384  # Token limit for embedding models
    ):
        """
        Initialize dataset with statements and document chunks.
        
        Args:
            statements_dir: Directory containing statement text files
            answers_dir: Directory containing answer JSON files
            topics_json_path: Path to topics.json mapping file
            topics_documents_dir: Directory containing topic folders with documents
            negative_ratio: Number of negatives per positive pair
            min_chunk_words: Minimum words per chunk
            max_chunk_words: Maximum words per chunk
            overlap_sentences: Number of sentences to overlap between chunks
            max_chunks_per_topic: Maximum chunks to use per topic
        """
        self.statements_dir = statements_dir
        self.answers_dir = answers_dir
        self.topics_documents_dir = topics_documents_dir
        self.negative_ratio = negative_ratio
        self.min_chunk_words = min_chunk_words
        self.max_chunk_words = max_chunk_words
        self.overlap_sentences = overlap_sentences
        self.max_chunks_per_topic = max_chunks_per_topic
        self.max_tokens_per_example = max_tokens_per_example
        
        # Load topic mapping
        self.topic_mapping = self._load_topic_mapping(topics_json_path)
        print(f"Loaded {len(self.topic_mapping)} topics")
        
        # Load statements and their topics
        self.statements, self.statement_topics = self._load_statements_and_topics()
        print(f"Loaded {len(self.statements)} statements")
        
        # Load all topic documents  
        self.documents_by_topic = self._load_and_chunk_documents()
        print(f"Loaded documents for {len(self.documents_by_topic)} topics")
        
        # Create training examples
        self.examples = self._create_training_examples()
        print(f"Created {len(self.examples)} training examples")
    
    def _load_topic_mapping(self, topics_json_path: str) -> Dict[int, str]:
        """Load topic ID to name mapping from topics.json."""
        with open(topics_json_path, 'r', encoding='utf-8') as f:
            topics_data = json.load(f)
        
        # Handle both formats: {name: id} or {id: name}
        topic_mapping = {}
        
        for key, value in topics_data.items():
            try:
                # Try to parse key as int (format: {"0": "name"})
                topic_id = int(key)
                topic_name = value
                topic_mapping[topic_id] = topic_name
            except ValueError:
                # Key is not int, so format is {"name": "id"}
                try:
                    topic_id = int(value)
                    topic_name = key
                    topic_mapping[topic_id] = topic_name
                except ValueError:
                    print(f"Warning: Could not parse topic mapping: {key} -> {value}")
        
        return topic_mapping
    
    def _load_statements_and_topics(self) -> Tuple[List[str], List[int]]:
        """Load statements and their corresponding topic IDs."""
        statements = []
        topics = []
        
        # Get all statement files
        statement_files = sorted(glob.glob(os.path.join(self.statements_dir, "*.txt")))
        
        for stmt_file in statement_files:
            # Extract file ID (e.g., statement_0001.txt -> 0001)
            file_id = os.path.basename(stmt_file).replace("statement_", "").replace(".txt", "")
            
            # Load statement text
            with open(stmt_file, 'r', encoding='utf-8') as f:
                statement_text = f.read().strip()
            
            # Load corresponding answer file
            answer_file = os.path.join(self.answers_dir, f"statement_{file_id}.json")
            if os.path.exists(answer_file):
                with open(answer_file, 'r', encoding='utf-8') as f:
                    answer_data = json.load(f)
                
                topic_id = answer_data.get("statement_topic")
                if topic_id is not None:
                    statements.append(statement_text)
                    topics.append(topic_id)
                else:
                    print(f"Warning: No topic_id found in {answer_file}")
            else:
                print(f"Warning: Answer file not found for {stmt_file}")
        
        return statements, topics
    
    def _load_and_chunk_documents(self) -> Dict[int, List[Dict[str, Any]]]:
        """Load and intelligently chunk documents by topic using advanced processing."""
        documents_by_topic = {}
        
        # Create reverse mapping: topic_name -> topic_id
        name_to_id = {name: topic_id for topic_id, name in self.topic_mapping.items()}
        
        # Walk through all topic folders in the topics directory
        for topic_folder_name in os.listdir(self.topics_documents_dir):
            topic_folder_path = os.path.join(self.topics_documents_dir, topic_folder_name)
            
            if not os.path.isdir(topic_folder_path):
                continue
            
            # Get topic ID from folder name
            topic_id = name_to_id.get(topic_folder_name)
            if topic_id is None:
                print(f"Warning: Topic folder '{topic_folder_name}' not found in mapping")
                continue
            
            # Get all markdown files in topic folder
            doc_files = glob.glob(os.path.join(topic_folder_path, "*.md"))
            
            all_chunks = []
            all_topic_content = []  # For combined content
            
            for doc_file in doc_files:
                try:
                    article_title = os.path.basename(doc_file).replace('.md', '')
                    
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        raw_content = f.read().strip()
                    
                    if raw_content:
                        # Clean the content using advanced cleaning
                        cleaned_content = clean_text(raw_content)
                        all_topic_content.append(cleaned_content)
                        
                        # Process into intelligent chunks
                        chunks = self._process_document_into_chunks(
                            cleaned_content, topic_id, topic_folder_name, article_title
                        )
                        all_chunks.extend(chunks)
                        
                except Exception as e:
                    print(f"Error processing {doc_file}: {e}")
            
            # Limit chunks per topic to avoid memory issues
            if len(all_chunks) > self.max_chunks_per_topic:
                all_chunks = random.sample(all_chunks, self.max_chunks_per_topic)
            
            if all_chunks:
                documents_by_topic[topic_id] = {
                    'chunks': all_chunks,
                    'combined_content': '\n\n'.join(all_topic_content),  # All articles combined
                    'chunk_count': len(all_chunks),
                    'doc_count': len(doc_files)
                }
                print(f"Topic {topic_id} ({topic_folder_name}): {len(all_chunks)} chunks from {len(doc_files)} documents")
        
        return documents_by_topic
    
    def _process_document_into_chunks(self, content: str, topic_id: int, topic_name: str, article_title: str) -> List[Dict[str, Any]]:
        """Process document into intelligent chunks with section awareness."""
        chunks = []
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
                        section_chunks = self._process_section(
                            section_text,
                            overlap_buffer,
                            topic_id,
                            topic_name,
                            article_title,
                            position_counter
                        )
                        chunks.extend(section_chunks)
                        position_counter += len(section_chunks)
                        
                        # Set overlap for next section from last chunk
                        if section_chunks:
                            overlap_buffer = get_overlap_sentences(section_chunks[-1]['content'])
                
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
                section_chunks = self._process_section(
                    section_text,
                    overlap_buffer,
                    topic_id,
                    topic_name,
                    article_title,
                    position_counter
                )
                chunks.extend(section_chunks)
        
        return chunks
    
    def _process_section(self, section_text: str, overlap_buffer: str, topic_id: int, 
                        topic_name: str, article_title: str, position_start: int) -> List[Dict[str, Any]]:
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
        
        if word_count <= self.max_chunk_words:
            # Section fits in one chunk - only create if meets minimum size
            if word_count >= self.min_chunk_words:
                # Normalize text
                normalized_text = unicodedata.normalize("NFKC", section_text).strip()
                chunks.append({
                    'content': normalized_text,
                    'metadata': {
                        'topic_id': topic_id,
                        'topic_name': topic_name,
                        'article_title': article_title,
                        'position': position_start,
                        'word_count': len(normalized_text.split())
                    }
                })
        else:
            # Need to split section into multiple chunks
            text_chunks = split_sentences_smart(section_text, self.max_chunk_words)
            
            for i, chunk_text in enumerate(text_chunks):
                chunk_word_count = len(chunk_text.split())
                
                # Add overlap from previous chunk (except for first chunk which already has it)
                if i > 0 and chunks:
                    prev_overlap = get_overlap_sentences(chunks[-1]['content'])
                    chunk_text = prev_overlap + " " + chunk_text
                    chunk_word_count = len(chunk_text.split())
                
                # Only add chunk if it meets minimum word count
                if chunk_word_count >= self.min_chunk_words:
                    # Normalize text
                    normalized_text = unicodedata.normalize("NFKC", chunk_text).strip()
                    chunks.append({
                        'content': normalized_text,
                        'metadata': {
                            'topic_id': topic_id,
                            'topic_name': topic_name,
                            'article_title': article_title,
                            'position': position_start + i,
                            'word_count': len(normalized_text.split())
                        }
                    })
        
        return chunks
    
    def _create_training_examples(self) -> List[InputExample]:
        """Create training examples optimized for embedding model fine-tuning."""
        examples = []
        
        for i, (statement, topic_id) in enumerate(zip(self.statements, self.statement_topics)):
            if topic_id in self.documents_by_topic:
                topic_data = self.documents_by_topic[topic_id]
                positive_chunks = topic_data['chunks']
                
                # Create multiple triplets per statement using different positive chunks
                # This addresses the token limit and provides more training diversity
                num_positives = min(3, len(positive_chunks))  # Use up to 3 different positive chunks
                selected_positives = random.sample(positive_chunks, num_positives)
                
                for pos_chunk in selected_positives:
                    # Ensure positive chunk is within token limit
                    pos_content = self._truncate_to_token_limit(pos_chunk['content'])
                    
                    # Create one positive example
                    examples.append(InputExample(
                        texts=[statement, pos_content],
                        label=1.0
                    ))
                    
                    # Create multiple negatives for this positive
                    negatives_added = 0
                    
                    # Add hard negatives (related medical topics)
                    negative_topics = self._get_related_topics(topic_id)
                    for neg_topic in negative_topics:
                        if negatives_added >= self.negative_ratio:
                            break
                        
                        if neg_topic in self.documents_by_topic:
                            neg_topic_data = self.documents_by_topic[neg_topic]
                            neg_chunks = neg_topic_data['chunks']
                            
                            # Select hard negative - chunk from related topic
                            if neg_chunks:
                                neg_chunk = random.choice(neg_chunks)
                                neg_content = self._truncate_to_token_limit(neg_chunk['content'])
                                
                                examples.append(InputExample(
                                    texts=[statement, neg_content],
                                    label=0.0
                                ))
                                negatives_added += 1
                    
                    # Add random negatives if needed
                    while negatives_added < self.negative_ratio:
                        available_topics = [t for t in self.documents_by_topic.keys() 
                                         if t != topic_id]
                        if available_topics:
                            random_topic = random.choice(available_topics)
                            random_topic_data = self.documents_by_topic[random_topic]
                            random_chunks = random_topic_data['chunks']
                            
                            if random_chunks:
                                random_chunk = random.choice(random_chunks)
                                neg_content = self._truncate_to_token_limit(random_chunk['content'])
                                
                                examples.append(InputExample(
                                    texts=[statement, neg_content],
                                    label=0.0
                                ))
                                negatives_added += 1
                        else:
                            break
            else:
                print(f"Warning: No documents found for topic {topic_id}")
        
        return examples
    
    def _truncate_to_token_limit(self, text: str) -> str:
        """Truncate text to stay within token limits (rough approximation)."""
        # Rough approximation: 1 token ≈ 0.75 words
        max_words = int(self.max_tokens_per_example * 0.75)
        words = text.split()
        
        if len(words) <= max_words:
            return text
        
        # Truncate at sentence boundary if possible
        truncated_text = ' '.join(words[:max_words])
        
        # Try to end at a sentence boundary
        last_period = truncated_text.rfind('.')
        if last_period > len(truncated_text) * 0.8:  # If period is in last 20%
            return truncated_text[:last_period + 1]
        
        return truncated_text
    
    def _get_related_topics(self, target_topic: int) -> List[int]:
        """Get related medical topics for hard negative sampling."""
        # Medical topic clustering based on clinical domains
        medical_clusters = {
            # Cardiovascular conditions
            frozenset([4, 7, 22, 23, 24, 25, 38, 49, 51, 57, 69, 77, 80, 82]),
            # Respiratory conditions  
            frozenset([8, 13, 14, 19, 21, 34, 45, 46, 47, 59, 61, 62, 63, 64, 65, 66, 67, 74, 81]),
            # Trauma and emergency conditions
            frozenset([0, 16, 20, 26, 28, 39, 55, 79]),
            # Gastrointestinal conditions
            frozenset([1, 2, 3, 17, 37, 54, 56]),
            # Neurological conditions  
            frozenset([18, 29, 35, 48, 71, 75, 76]),
            # Infectious diseases and sepsis
            frozenset([36, 72, 61, 48, 35]),
            # Metabolic and endocrine
            frozenset([30, 42, 43, 44, 6, 5]),
            # Reproductive and gynecological
            frozenset([31, 32, 52, 58, 78]),
            # Diagnostic tests and procedures
            frozenset(range(83, 115))  # Topics 83-114
        }
        
        # Find cluster containing target topic
        target_cluster = None
        for cluster in medical_clusters:
            if target_topic in cluster:
                target_cluster = cluster
                break
        
        if target_cluster:
            # Return other topics in same cluster (excluding target)
            related = list(target_cluster - {target_topic})
            # Prioritize topics that actually have documents
            available_related = [t for t in related if t in self.documents_by_topic]
            return available_related[:5]  # Return top 5 related topics
        else:
            # Return random topics if not in any cluster
            all_topics = list(self.documents_by_topic.keys())
            available = [t for t in all_topics if t != target_topic]
            return random.sample(available, min(5, len(available)))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        positive_examples = sum(1 for ex in self.examples if ex.label == 1.0)
        negative_examples = sum(1 for ex in self.examples if ex.label == 0.0)
        
        return {
            'total_examples': len(self.examples),
            'positive_examples': positive_examples,
            'negative_examples': negative_examples,
            'positive_ratio': positive_examples / len(self.examples) if self.examples else 0,
            'total_statements': len(self.statements),
            'total_topics': len(self.documents_by_topic),
            'chunks_per_topic': {
                topic_id: topic_data['chunk_count'] 
                for topic_id, topic_data in self.documents_by_topic.items()
            },
            'docs_per_topic': {
                topic_id: topic_data['doc_count'] 
                for topic_id, topic_data in self.documents_by_topic.items()
            }
        }
    
    def save_to_json(self, output_path: str):
        """Save the dataset optimized for embedding model training."""
        examples = []
        
        # Create triplet format optimized for embedding training
        for ex in self.examples:
            if ex.label == 1.0:  # This is an anchor-positive pair
                anchor = ex.texts[0]
                positive = ex.texts[1]
                
                # Find negatives for this anchor
                negatives = []
                for neg_ex in self.examples:
                    if (neg_ex.label == 0.0 and 
                        neg_ex.texts[0] == anchor and 
                        len(negatives) < self.negative_ratio):
                        negatives.append(neg_ex.texts[1])
                
                # Only include if we have negatives
                if negatives:
                    examples.append({
                        'anchor': anchor,
                        'positive': positive,
                        'negatives': negatives
                    })
        
        # Add metadata about token lengths
        token_stats = {
            'avg_anchor_tokens': self._estimate_avg_tokens([ex['anchor'] for ex in examples]),
            'avg_positive_tokens': self._estimate_avg_tokens([ex['positive'] for ex in examples]),
            'avg_negative_tokens': self._estimate_avg_tokens([neg for ex in examples for neg in ex['negatives']]),
            'max_tokens_limit': self.max_tokens_per_example
        }
        
        data = {
            'examples': examples,
            'token_stats': token_stats,
            'training_stats': self.get_stats()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to {output_path}")
        print(f"Token statistics: {token_stats}")
    
    def _estimate_avg_tokens(self, texts):
        """Estimate average token count for a list of texts."""
        if not texts:
            return 0
        total_words = sum(len(text.split()) for text in texts)
        avg_words = total_words / len(texts)
        return int(avg_words / 0.75)  # Convert words to approximate tokens


def create_medical_dataset(
    statements_dir: str = "data/processed/combined_train/statements",
    answers_dir: str = "data/processed/combined_train/answers", 
    topics_json_path: str = "data/topics.json",
    topics_documents_dir: str = "data/raw/topics",
    **kwargs
) -> MedicalTripletDataset:
    """
    Convenience function to create a medical dataset with default paths.
    
    Args:
        statements_dir: Directory containing statement files
        answers_dir: Directory containing answer files
        topics_json_path: Path to topics.json
        topics_documents_dir: Directory containing topic document folders
        **kwargs: Additional arguments for MedicalTripletDataset
    
    Returns:
        MedicalTripletDataset instance
    """
    return MedicalTripletDataset(
        statements_dir=statements_dir,
        answers_dir=answers_dir,
        topics_json_path=topics_json_path,
        topics_documents_dir=topics_documents_dir,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    # Create dataset
    print("Creating medical triplet dataset...")
    
    try:
        dataset = create_medical_dataset(
            negative_ratio=4,  # More negatives for better learning
            min_chunk_words=50,  # Smaller chunks
            max_chunk_words=150,  # Fit in token limits
            overlap_sentences=1,  # Less overlap
            max_chunks_per_topic=20,  # More variety
            max_tokens_per_example=384  # BERT token limit
        )
        
        # Print statistics
        stats = dataset.get_stats()
        print("\nDataset Statistics:")
        print(f"Total examples: {stats['total_examples']}")
        print(f"Positive examples: {stats['positive_examples']}")
        print(f"Negative examples: {stats['negative_examples']}")
        print(f"Positive ratio: {stats['positive_ratio']:.2%}")
        print(f"Total statements: {stats['total_statements']}")
        print(f"Total topics with documents: {stats['total_topics']}")
        print(f"Total chunks created: {sum(stats['chunks_per_topic'].values())}")
        print(f"Average chunks per topic: {sum(stats['chunks_per_topic'].values()) / len(stats['chunks_per_topic']):.1f}")
        
        # Show some examples from the saved format
        print("\nPreview of optimized triplet format:")
        # Find first few anchor-positive pairs
        positive_examples = [ex for ex in dataset.examples if ex.label == 1.0][:3]
        
        for i, pos_ex in enumerate(positive_examples):
            anchor = pos_ex.texts[0]
            positive = pos_ex.texts[1]
            
            # Find negatives for this anchor
            negatives = []
            for neg_ex in dataset.examples:
                if (neg_ex.label == 0.0 and neg_ex.texts[0] == anchor and len(negatives) < 2):
                    negatives.append(neg_ex.texts[1])
            
            print(f"\n--- Triplet {i+1} ---")
            print(f"Anchor: {anchor[:100]}...")
            print(f"Positive ({len(positive.split())} words, ~{int(len(positive.split())/0.75)} tokens): {positive[:120]}...")
            if negatives:
                print(f"Negatives ({len(negatives)} total):")
                for j, neg in enumerate(negatives[:2]):
                    print(f"  Neg {j+1} ({len(neg.split())} words, ~{int(len(neg.split())/0.75)} tokens): {neg[:100]}...")
        
        # Save dataset for later use
        dataset.save_to_json("medical_embedding_dataset.json")
        print("\n[SUCCESS] Dataset creation completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Error creating dataset: {e}")
        import traceback
        traceback.print_exc()