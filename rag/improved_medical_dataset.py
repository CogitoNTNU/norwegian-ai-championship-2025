"""
Improved medical embedding dataset generation with semantic context selection.
Addresses issues with random chunk selection by using semantic similarity and full article context.
"""

import os
import json
import re
import unicodedata
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import glob
from pathlib import Path
from datetime import datetime
import logging

# Text processing imports
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    try:
        sent_tokenize("Test.")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            nltk.download("punkt", quiet=True)
except ImportError:
    def sent_tokenize(text):
        sentences = re.split(r"\.[\s\n]+|\.$", text)
        return [s.strip() + "." for s in sentences if s.strip()]

# Semantic similarity imports
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_MODEL = None
    SEMANTIC_AVAILABLE = False
    print("Warning: sentence-transformers not available. Using fallback text matching.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Text cleaning (reuse from original)
link_re = re.compile(r"\[([^\]]+?)\]\([^)]+?\)")

def clean_text(raw):
    """Comprehensive text cleaning with residual boilerplate removal."""
    lines = raw.split("\n")
    cleaned_lines = []
    in_references = False

    for line in lines:
        if re.match(r"^[#*\s]*References\s*$", line.strip(), re.IGNORECASE):
            in_references = True
            break
        if in_references:
            break

        if (
            line.strip().startswith("## source:")
            or line.strip().startswith("**Disclosure:")
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
    txt = re.sub(r"<https?://[^>]+>", "", txt)
    txt = re.sub(r"\[Level\s*\d+\]", "", txt)
    txt = re.sub(r"\[\d+\](\[\d+\])*", "", txt)
    txt = re.sub(r"\[\s*\]\(\s*\)", "", txt)
    txt = txt.replace("\\*", "").replace("\\\\", "")

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

    txt = re.sub(r"\n{2,}", "\n", txt).strip()
    return txt

def compute_semantic_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity between two texts."""
    if not SEMANTIC_AVAILABLE:
        # Fallback: simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0
    
    try:
        embeddings = SEMANTIC_MODEL.encode([text1, text2])
        similarity = float(np.dot(embeddings[0], embeddings[1]) / 
                          (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
        return max(0, similarity)  # Ensure non-negative
    except Exception:
        # Fallback to word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0

def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 0.75 words)."""
    return int(len(text.split()) / 0.75)

class ImprovedMedicalDataset:
    """Improved medical dataset with semantic context selection."""
    
    def __init__(
        self,
        statements_dir: str,
        answers_dir: str,
        topics_json_path: str,
        topics_documents_dir: str,
        negative_ratio: int = 3,
        max_tokens_per_example: int = 512,  # Increased token limit
        use_full_articles: bool = True,      # NEW: Use full articles when possible
        multi_chunk_context: bool = True,   # NEW: Combine multiple relevant chunks
        semantic_threshold: float = 0.3,    # NEW: Minimum similarity for relevance
        max_articles_per_topic: int = 5     # NEW: Limit articles per topic for full context
    ):
        """
        Initialize improved dataset with semantic context selection.
        
        Args:
            statements_dir: Directory containing statement text files
            answers_dir: Directory containing answer JSON files
            topics_json_path: Path to topics.json mapping file
            topics_documents_dir: Directory containing topic folders with documents
            negative_ratio: Number of negatives per positive pair
            max_tokens_per_example: Maximum tokens per example
            use_full_articles: Whether to use full articles as context when possible
            multi_chunk_context: Whether to combine multiple relevant chunks
            semantic_threshold: Minimum semantic similarity for relevance
            max_articles_per_topic: Maximum articles to combine per topic
        """
        self.statements_dir = statements_dir
        self.answers_dir = answers_dir
        self.topics_documents_dir = topics_documents_dir
        self.negative_ratio = negative_ratio
        self.max_tokens_per_example = max_tokens_per_example
        self.use_full_articles = use_full_articles
        self.multi_chunk_context = multi_chunk_context
        self.semantic_threshold = semantic_threshold
        self.max_articles_per_topic = max_articles_per_topic
        
        logger.info(f"Initializing improved dataset:")
        logger.info(f"  Max tokens per example: {max_tokens_per_example}")
        logger.info(f"  Use full articles: {use_full_articles}")
        logger.info(f"  Multi-chunk context: {multi_chunk_context}")
        logger.info(f"  Semantic threshold: {semantic_threshold}")
        logger.info(f"  Semantic similarity: {'Available' if SEMANTIC_AVAILABLE else 'Fallback mode'}")
        
        # Load data
        self.topic_mapping = self._load_topic_mapping(topics_json_path)
        self.statements, self.statement_topics = self._load_statements_and_topics()
        self.documents_by_topic = self._load_topic_documents()
        self.examples = self._create_improved_examples()
        
        logger.info(f"Dataset created: {len(self.examples)} examples")
    
    def _load_topic_mapping(self, topics_json_path: str) -> Dict[int, str]:
        """Load topic ID to name mapping."""
        with open(topics_json_path, 'r', encoding='utf-8') as f:
            topics_data = json.load(f)
        
        topic_mapping = {}
        for key, value in topics_data.items():
            try:
                topic_id = int(key)
                topic_name = value
                topic_mapping[topic_id] = topic_name
            except ValueError:
                try:
                    topic_id = int(value)
                    topic_name = key
                    topic_mapping[topic_id] = topic_name
                except ValueError:
                    logger.warning(f"Could not parse topic mapping: {key} -> {value}")
        
        logger.info(f"Loaded {len(topic_mapping)} topics")
        return topic_mapping
    
    def _load_statements_and_topics(self) -> Tuple[List[str], List[int]]:
        """Load statements and their corresponding topic IDs."""
        statements = []
        topics = []
        
        statement_files = sorted(glob.glob(os.path.join(self.statements_dir, "*.txt")))
        
        for stmt_file in statement_files:
            file_id = os.path.basename(stmt_file).replace("statement_", "").replace(".txt", "")
            
            with open(stmt_file, 'r', encoding='utf-8') as f:
                statement_text = f.read().strip()
            
            answer_file = os.path.join(self.answers_dir, f"statement_{file_id}.json")
            if os.path.exists(answer_file):
                with open(answer_file, 'r', encoding='utf-8') as f:
                    answer_data = json.load(f)
                
                topic_id = answer_data.get("statement_topic")
                if topic_id is not None:
                    statements.append(statement_text)
                    topics.append(topic_id)
        
        logger.info(f"Loaded {len(statements)} statements")
        return statements, topics
    
    def _load_topic_documents(self) -> Dict[int, Dict[str, Any]]:
        """Load and organize documents by topic with full article preservation."""
        documents_by_topic = {}
        name_to_id = {name: topic_id for topic_id, name in self.topic_mapping.items()}
        
        for topic_folder_name in os.listdir(self.topics_documents_dir):
            topic_folder_path = os.path.join(self.topics_documents_dir, topic_folder_name)
            
            if not os.path.isdir(topic_folder_path):
                continue
            
            topic_id = name_to_id.get(topic_folder_name)
            if topic_id is None:
                continue
            
            doc_files = glob.glob(os.path.join(topic_folder_path, "*.md"))
            
            articles = []
            all_content = []
            
            for doc_file in doc_files:
                try:
                    article_title = os.path.basename(doc_file).replace('.md', '')
                    
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        raw_content = f.read().strip()
                    
                    if raw_content:
                        cleaned_content = clean_text(raw_content)
                        token_count = estimate_tokens(cleaned_content)
                        
                        articles.append({
                            'title': article_title,
                            'content': cleaned_content,
                            'token_count': token_count,
                            'file_path': doc_file
                        })
                        all_content.append(cleaned_content)
                        
                except Exception as e:
                    logger.warning(f"Error processing {doc_file}: {e}")
            
            if articles:
                # Sort articles by relevance/length for better context selection
                articles.sort(key=lambda x: x['token_count'], reverse=True)
                
                # Combine content strategically
                combined_content = self._create_combined_content(articles)
                
                documents_by_topic[topic_id] = {
                    'articles': articles,
                    'combined_content': combined_content,
                    'total_articles': len(articles),
                    'total_tokens': sum(a['token_count'] for a in articles)
                }
                
                logger.info(f"Topic {topic_id}: {len(articles)} articles, "
                          f"{sum(a['token_count'] for a in articles)} total tokens")
        
        logger.info(f"Loaded documents for {len(documents_by_topic)} topics")
        return documents_by_topic
    
    def _create_combined_content(self, articles: List[Dict[str, Any]]) -> str:
        """Create intelligently combined content from multiple articles."""
        if not articles:
            return ""
        
        if len(articles) == 1:
            return articles[0]['content']
        
        # If using full articles and they fit within token limit
        if self.use_full_articles:
            combined = []
            total_tokens = 0
            
            for article in articles[:self.max_articles_per_topic]:
                if total_tokens + article['token_count'] <= self.max_tokens_per_example:
                    combined.append(f"# {article['title']}\n\n{article['content']}")
                    total_tokens += article['token_count']
                else:
                    # Add partial content if possible
                    remaining_tokens = self.max_tokens_per_example - total_tokens
                    if remaining_tokens > 100:  # Only if significant space left
                        partial_content = self._truncate_content(
                            article['content'], remaining_tokens
                        )
                        combined.append(f"# {article['title']}\n\n{partial_content}")
                    break
            
            if combined:
                return "\n\n".join(combined)
        
        # Fallback: use the most comprehensive single article
        return articles[0]['content']
    
    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Intelligently truncate content to fit token limit."""
        max_words = int(max_tokens * 0.75)
        words = content.split()
        
        if len(words) <= max_words:
            return content
        
        # Try to truncate at sentence boundary
        truncated_text = ' '.join(words[:max_words])
        sentences = sent_tokenize(truncated_text)
        
        if len(sentences) > 1:
            # Remove potentially incomplete last sentence
            return ' '.join(sentences[:-1])
        
        return truncated_text
    
    def _create_improved_examples(self) -> List[Dict[str, Any]]:
        """Create training examples with improved context selection."""
        examples = []
        
        for i, (statement, topic_id) in enumerate(zip(self.statements, self.statement_topics)):
            if topic_id not in self.documents_by_topic:
                logger.warning(f"No documents found for topic {topic_id}")
                continue
            
            topic_data = self.documents_by_topic[topic_id]
            
            # NEW: Semantic-based context selection
            positive_context = self._select_best_context(statement, topic_data)
            
            if not positive_context:
                logger.warning(f"No suitable context found for statement {i}")
                continue
            
            # Create positive example
            positive_example = {
                'anchor': statement,
                'positive': positive_context,
                'negatives': []
            }
            
            # Create negatives using improved selection
            negatives = self._select_negatives(statement, topic_id)
            positive_example['negatives'] = negatives
            
            if negatives:  # Only add if we have negatives
                examples.append(positive_example)
                
                # Log example quality for first few examples
                if i < 3:
                    similarity = compute_semantic_similarity(statement, positive_context)
                    logger.info(f"Example {i+1}:")
                    logger.info(f"  Statement: {statement[:100]}...")
                    logger.info(f"  Similarity: {similarity:.3f}")
                    logger.info(f"  Context tokens: {estimate_tokens(positive_context)}")
                    logger.info(f"  Negatives: {len(negatives)}")
        
        return examples
    
    def _select_best_context(self, statement: str, topic_data: Dict[str, Any]) -> str:
        """Select the most relevant context for a statement using semantic similarity."""
        articles = topic_data['articles']
        
        if not articles:
            return ""
        
        # Option 1: Use combined content if it fits
        combined_content = topic_data['combined_content']
        if estimate_tokens(combined_content) <= self.max_tokens_per_example:
            similarity = compute_semantic_similarity(statement, combined_content)
            if similarity >= self.semantic_threshold:
                logger.debug(f"Using combined content (similarity: {similarity:.3f})")
                return combined_content
        
        # Option 2: Find best single article
        best_article = None
        best_similarity = 0.0
        
        for article in articles:
            if estimate_tokens(article['content']) > self.max_tokens_per_example:
                # Truncate large articles
                truncated_content = self._truncate_content(
                    article['content'], self.max_tokens_per_example
                )
                similarity = compute_semantic_similarity(statement, truncated_content)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_article = truncated_content
            else:
                similarity = compute_semantic_similarity(statement, article['content'])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_article = article['content']
        
        if best_article and best_similarity >= self.semantic_threshold:
            logger.debug(f"Using best article (similarity: {best_similarity:.3f})")
            return best_article
        
        # Option 3: Multi-chunk approach if enabled
        if self.multi_chunk_context:
            return self._create_multi_chunk_context(statement, articles)
        
        # Fallback: use the longest article (truncated if necessary)
        fallback_article = articles[0]['content']
        if estimate_tokens(fallback_article) > self.max_tokens_per_example:
            fallback_article = self._truncate_content(
                fallback_article, self.max_tokens_per_example
            )
        
        logger.debug("Using fallback article")
        return fallback_article
    
    def _create_multi_chunk_context(self, statement: str, articles: List[Dict[str, Any]]) -> str:
        """Create context by combining the most relevant sections from multiple articles."""
        # Split articles into sections and find most relevant ones
        relevant_sections = []
        
        for article in articles:
            sections = self._split_into_sections(article['content'])
            
            for section in sections:
                if len(section.split()) > 50:  # Minimum section size
                    similarity = compute_semantic_similarity(statement, section)
                    if similarity >= self.semantic_threshold:
                        relevant_sections.append({
                            'content': section,
                            'similarity': similarity,
                            'article_title': article['title']
                        })
        
        if not relevant_sections:
            return ""
        
        # Sort by similarity and combine top sections
        relevant_sections.sort(key=lambda x: x['similarity'], reverse=True)
        
        combined_sections = []
        total_tokens = 0
        
        for section in relevant_sections:
            section_tokens = estimate_tokens(section['content'])
            if total_tokens + section_tokens <= self.max_tokens_per_example:
                combined_sections.append(section['content'])
                total_tokens += section_tokens
            else:
                break
        
        if combined_sections:
            return "\n\n".join(combined_sections)
        
        return ""
    
    def _split_into_sections(self, content: str) -> List[str]:
        """Split content into meaningful sections."""
        # Split by headings or paragraphs
        sections = []
        current_section = []
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Check if this is a heading
            if re.match(r'^#+\s', line) and current_section:
                # Save current section and start new one
                section_text = '\n'.join(current_section).strip()
                if section_text:
                    sections.append(section_text)
                current_section = [line]
            else:
                current_section.append(line)
        
        # Add final section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if section_text:
                sections.append(section_text)
        
        return sections
    
    def _select_negatives(self, statement: str, positive_topic_id: int) -> List[str]:
        """Select high-quality negative examples."""
        negatives = []
        
        # Get all available topics except the positive one
        available_topics = [tid for tid in self.documents_by_topic.keys() 
                          if tid != positive_topic_id]
        
        # Add some hard negatives (related medical topics) and random negatives
        selected_topics = available_topics[:self.negative_ratio * 2]  # Get more candidates
        
        for topic_id in selected_topics:
            if len(negatives) >= self.negative_ratio:
                break
            
            topic_data = self.documents_by_topic[topic_id]
            
            # Select negative context from this topic
            if topic_data['articles']:
                # Use first article as negative (could be improved with anti-similarity)
                negative_article = topic_data['articles'][0]
                negative_content = negative_article['content']
                
                # Truncate if necessary
                if estimate_tokens(negative_content) > self.max_tokens_per_example:
                    negative_content = self._truncate_content(
                        negative_content, self.max_tokens_per_example
                    )
                
                negatives.append(negative_content)
        
        return negatives
    
    def save_to_json(self, output_path: str):
        """Save the improved dataset to JSON."""
        # Calculate statistics
        token_stats = {
            'avg_anchor_tokens': np.mean([estimate_tokens(ex['anchor']) for ex in self.examples]),
            'avg_positive_tokens': np.mean([estimate_tokens(ex['positive']) for ex in self.examples]),
            'avg_negative_tokens': np.mean([
                estimate_tokens(neg) for ex in self.examples for neg in ex['negatives']
            ]),
            'max_tokens_limit': self.max_tokens_per_example,
            'semantic_similarity_used': SEMANTIC_AVAILABLE
        }
        
        # Calculate semantic quality
        if SEMANTIC_AVAILABLE:
            similarities = [
                compute_semantic_similarity(ex['anchor'], ex['positive']) 
                for ex in self.examples
            ]
            token_stats['avg_semantic_similarity'] = float(np.mean(similarities))
            token_stats['min_semantic_similarity'] = float(np.min(similarities))
        
        data = {
            'examples': self.examples,
            'generation_config': {
                'use_full_articles': self.use_full_articles,
                'multi_chunk_context': self.multi_chunk_context,
                'semantic_threshold': self.semantic_threshold,
                'max_tokens_per_example': self.max_tokens_per_example,
                'negative_ratio': self.negative_ratio,
                'generation_timestamp': datetime.now().isoformat()
            },
            'token_stats': token_stats,
            'dataset_stats': {
                'total_examples': len(self.examples),
                'total_statements': len(self.statements),
                'total_topics': len(self.documents_by_topic)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Improved dataset saved to {output_path}")
        logger.info(f"Token statistics: {token_stats}")

def create_improved_medical_dataset(**kwargs) -> ImprovedMedicalDataset:
    """Create improved medical dataset with better context selection."""
    default_paths = {
        'statements_dir': "data/processed/combined_train/statements",
        'answers_dir': "data/processed/combined_train/answers",
        'topics_json_path': "data/topics.json",
        'topics_documents_dir': "data/raw/topics"
    }
    
    # Update with user provided kwargs
    config = {**default_paths, **kwargs}
    
    return ImprovedMedicalDataset(**config)

if __name__ == "__main__":
    print("Creating improved medical embedding dataset...")
    print("=" * 60)
    
    try:
        # Create improved dataset with better configurations
        dataset = create_improved_medical_dataset(
            negative_ratio=3,
            max_tokens_per_example=512,     # Increased token limit
            use_full_articles=True,         # Enable full article context
            multi_chunk_context=True,       # Enable multi-chunk combination
            semantic_threshold=0.2,         # Lower threshold for more examples
            max_articles_per_topic=3        # Combine up to 3 articles per topic
        )
        
        # Show preview
        if dataset.examples:
            print(f"\nCreated {len(dataset.examples)} examples")
            print("\nPreview of improved examples:")
            print("-" * 60)
            
            for i, example in enumerate(dataset.examples[:2]):
                print(f"\nExample {i+1}:")
                print(f"Statement: {example['anchor'][:120]}...")
                print(f"Context tokens: {estimate_tokens(example['positive'])}")
                print(f"Context preview: {example['positive'][:200]}...")
                print(f"Negatives: {len(example['negatives'])}")
                
                if SEMANTIC_AVAILABLE:
                    similarity = compute_semantic_similarity(
                        example['anchor'], example['positive']
                    )
                    print(f"Semantic similarity: {similarity:.3f}")
        
        # Save improved dataset
        dataset.save_to_json("improved_medical_embedding_dataset.json")
        print(f"\n✅ SUCCESS: Improved dataset saved!")
        print("Key improvements:")
        print("  ✓ Semantic similarity-based context selection")
        print("  ✓ Full article context when possible")
        print("  ✓ Multi-chunk context combination")
        print("  ✓ Increased token limits (512 tokens)")
        print("  ✓ Better medical concept preservation")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()