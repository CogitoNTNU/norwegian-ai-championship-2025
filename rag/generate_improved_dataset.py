"""
Generate improved medical embedding dataset with semantic context selection.
Works without heavy dependencies that have version conflicts.
"""

import os
import json
import re
import unicodedata
import random
from typing import List, Dict, Any, Tuple, Optional
import glob
from pathlib import Path
from datetime import datetime
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Text processing functions
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

def clean_text(raw):
    """Comprehensive text cleaning with medical content preservation."""
    lines = raw.split("\n")
    cleaned_lines = []
    in_references = False
    
    for line in lines:
        # Stop at References section
        if re.match(r"^[#*\s]*References\s*$", line.strip(), re.IGNORECASE):
            in_references = True
            break
        if in_references:
            break
        
        # Skip boilerplate and metadata
        if (
            line.strip().startswith("## source:")
            or line.strip().startswith("**Disclosure:")
            or line.strip().startswith("Author Information")
            or line.strip().startswith("#### Authors")
            or line.strip().startswith("_" * 10)  # Separator lines
            or not line.strip()
        ):
            continue
        
        # Skip promotional content
        promo = line.lower()
        if (
            promo.startswith("access free multiple choice")
            or "comment on this article" in promo
            or promo.startswith("continuing education activity")
        ):
            continue
        
        cleaned_lines.append(line)
    
    # Join and normalize
    txt = "\n".join(cleaned_lines)
    txt = unicodedata.normalize("NFKC", txt)
    
    # Remove markdown links but keep text
    txt = re.sub(r"\[([^\]]+?)\]\([^)]+?\)", r"\1", txt)
    
    # Remove URLs
    txt = re.sub(r"<https?://[^>]+>", "", txt)
    
    # Remove evidence levels
    txt = re.sub(r"\[Level\s*\d+\]", "", txt)
    
    # Remove numeric citations
    txt = re.sub(r"\[\d+\](\[\d+\])*", "", txt)
    
    # Remove empty links
    txt = re.sub(r"\[\s*\]\(\s*\)", "", txt)
    
    # Clean up markdown artifacts
    txt = txt.replace("\\*", "").replace("\\\\", "")
    
    # Remove extra whitespace
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    
    return txt

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity using advanced text overlap with medical term weighting.
    """
    def extract_medical_terms(text):
        """Extract medical terms and important words."""
        # Common medical prefixes/suffixes that indicate important terms
        medical_patterns = [
            r'\b\w*cardio\w*\b',     # cardiovascular terms
            r'\b\w*pulmon\w*\b',     # pulmonary terms  
            r'\b\w*emia\b',          # blood conditions
            r'\b\w*osis\b',          # disease conditions
            r'\b\w*itis\b',          # inflammatory conditions
            r'\b\w*pathy\b',         # disease terms
            r'\b\w*therapy\b',       # treatment terms
            r'\b\w*syndrome\b',      # syndrome terms
        ]
        
        words = text.lower().split()
        medical_terms = set()
        regular_words = set()
        
        for word in words:
            word = re.sub(r'[^\w]', '', word)  # Clean punctuation
            if len(word) < 3:
                continue
            
            is_medical = False
            for pattern in medical_patterns:
                if re.search(pattern, word, re.IGNORECASE):
                    medical_terms.add(word)
                    is_medical = True
                    break
            
            if not is_medical:
                # Check if it's a likely medical term (longer, contains specific patterns)
                if (len(word) > 6 and 
                    (word.endswith('al') or word.endswith('ic') or word.endswith('tion'))):
                    medical_terms.add(word)
                else:
                    regular_words.add(word)
        
        return medical_terms, regular_words
    
    # Extract terms from both texts
    med1, reg1 = extract_medical_terms(text1)
    med2, reg2 = extract_medical_terms(text2)
    
    # Calculate weighted similarity
    medical_intersection = med1.intersection(med2)
    medical_union = med1.union(med2)
    
    regular_intersection = reg1.intersection(reg2)
    regular_union = reg1.union(reg2)
    
    # Medical terms get 3x weight, regular words get 1x weight
    medical_similarity = len(medical_intersection) / len(medical_union) if medical_union else 0
    regular_similarity = len(regular_intersection) / len(regular_union) if regular_union else 0
    
    # Weighted average: medical terms are 3x more important
    total_weight = 3 * len(medical_union) + len(regular_union)
    if total_weight == 0:
        return 0.0
    
    weighted_similarity = (3 * len(medical_union) * medical_similarity + 
                          len(regular_union) * regular_similarity) / total_weight
    
    return min(1.0, weighted_similarity)

def estimate_tokens(text: str) -> int:
    """Estimate token count (1 token ≈ 0.75 words for medical text)."""
    return int(len(text.split()) / 0.75)

class ImprovedMedicalDatasetGenerator:
    """Generate improved medical dataset with semantic context selection."""
    
    def __init__(
        self,
        statements_dir: str = "data/processed/combined_train/statements",
        answers_dir: str = "data/processed/combined_train/answers", 
        topics_json_path: str = "data/topics.json",
        topics_documents_dir: str = "data/raw/topics",
        max_tokens_per_example: int = 512,
        use_full_articles: bool = True,
        semantic_threshold: float = 0.1,  # Lower threshold for more matches
        negative_ratio: int = 3,
        max_articles_per_topic: int = 3
    ):
        """Initialize the improved dataset generator."""
        self.statements_dir = statements_dir
        self.answers_dir = answers_dir
        self.topics_json_path = topics_json_path
        self.topics_documents_dir = topics_documents_dir
        self.max_tokens_per_example = max_tokens_per_example
        self.use_full_articles = use_full_articles
        self.semantic_threshold = semantic_threshold
        self.negative_ratio = negative_ratio
        self.max_articles_per_topic = max_articles_per_topic
        
        logger.info("Initializing improved dataset generator...")
        logger.info(f"  Max tokens: {max_tokens_per_example}")
        logger.info(f"  Use full articles: {use_full_articles}")
        logger.info(f"  Semantic threshold: {semantic_threshold}")
        
        # Load all data
        self.topic_mapping = self._load_topic_mapping()
        self.statements, self.statement_topics = self._load_statements()
        self.documents_by_topic = self._load_documents()
        
    def _load_topic_mapping(self) -> Dict[int, str]:
        """Load topic ID to name mapping."""
        with open(self.topics_json_path, 'r', encoding='utf-8') as f:
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
                    continue
        
        logger.info(f"Loaded {len(topic_mapping)} topics")
        return topic_mapping
    
    def _load_statements(self) -> Tuple[List[str], List[int]]:
        """Load statements and their topic IDs."""
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
    
    def _load_documents(self) -> Dict[int, Dict[str, Any]]:
        """Load and process documents by topic."""
        documents_by_topic = {}
        name_to_id = {name: topic_id for topic_id, name in self.topic_mapping.items()}
        
        for topic_folder in os.listdir(self.topics_documents_dir):
            topic_path = os.path.join(self.topics_documents_dir, topic_folder)
            
            if not os.path.isdir(topic_path):
                continue
            
            topic_id = name_to_id.get(topic_folder)
            if topic_id is None:
                continue
            
            # Load articles for this topic
            articles = []
            doc_files = glob.glob(os.path.join(topic_path, "*.md"))
            
            for doc_file in doc_files:
                try:
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        raw_content = f.read().strip()
                    
                    if raw_content:
                        cleaned = clean_text(raw_content)
                        if cleaned:
                            articles.append({
                                'title': os.path.basename(doc_file).replace('.md', ''),
                                'content': cleaned,
                                'token_count': estimate_tokens(cleaned),
                                'path': doc_file
                            })
                except Exception as e:
                    logger.warning(f"Error processing {doc_file}: {e}")
            
            if articles:
                # Sort by token count (longer articles first)
                articles.sort(key=lambda x: x['token_count'], reverse=True)
                
                documents_by_topic[topic_id] = {
                    'articles': articles,
                    'total_articles': len(articles),
                    'total_tokens': sum(a['token_count'] for a in articles)
                }
                
                logger.info(f"Topic {topic_id}: {len(articles)} articles")
        
        logger.info(f"Loaded documents for {len(documents_by_topic)} topics")
        return documents_by_topic
    
    def _select_best_context(self, statement: str, topic_data: Dict[str, Any]) -> Tuple[str, float]:
        """Select the best context for a statement using semantic similarity."""
        articles = topic_data['articles']
        
        if not articles:
            return "", 0.0
        
        best_context = ""
        best_similarity = 0.0
        
        # Strategy 1: Try to use full article if it fits
        if self.use_full_articles:
            for article in articles[:self.max_articles_per_topic]:
                content = article['content']
                
                if estimate_tokens(content) <= self.max_tokens_per_example:
                    similarity = calculate_text_similarity(statement, content)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_context = content
                        logger.debug(f"Full article selected (sim: {similarity:.3f})")
                
                # Try combining multiple articles
                if best_similarity < self.semantic_threshold and len(articles) > 1:
                    combined_content = self._combine_articles(articles[:self.max_articles_per_topic])
                    if estimate_tokens(combined_content) <= self.max_tokens_per_example:
                        similarity = calculate_text_similarity(statement, combined_content)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_context = combined_content
                            logger.debug(f"Combined articles selected (sim: {similarity:.3f})")
        
        # Strategy 2: Smart truncation of best matching article
        if best_similarity < self.semantic_threshold:
            for article in articles:
                truncated = self._smart_truncate(article['content'], statement)
                similarity = calculate_text_similarity(statement, truncated)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_context = truncated
                    logger.debug(f"Truncated article selected (sim: {similarity:.3f})")
        
        # Strategy 3: Fallback to best section
        if best_similarity < self.semantic_threshold:
            best_section = self._find_best_section(statement, articles)
            if best_section:
                similarity = calculate_text_similarity(statement, best_section)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_context = best_section
                    logger.debug(f"Best section selected (sim: {similarity:.3f})")
        
        return best_context, best_similarity
    
    def _combine_articles(self, articles: List[Dict[str, Any]]) -> str:
        """Combine multiple articles intelligently."""
        combined_parts = []
        total_tokens = 0
        
        for article in articles:
            content = article['content']
            tokens = estimate_tokens(content)
            
            if total_tokens + tokens <= self.max_tokens_per_example:
                combined_parts.append(f"## {article['title']}\n\n{content}")
                total_tokens += tokens
            else:
                # Add partial content
                remaining_tokens = self.max_tokens_per_example - total_tokens
                if remaining_tokens > 100:  # Only if significant space
                    partial = self._smart_truncate(content, "", max_tokens=remaining_tokens)
                    combined_parts.append(f"## {article['title']}\n\n{partial}")
                break
        
        return "\n\n".join(combined_parts)
    
    def _smart_truncate(self, content: str, query: str = "", max_tokens: int = None) -> str:
        """Smart truncation preserving relevant content."""
        if max_tokens is None:
            max_tokens = self.max_tokens_per_example
        
        if estimate_tokens(content) <= max_tokens:
            return content
        
        # Split into sentences
        sentences = sent_tokenize(content)
        
        if query:
            # Score sentences by relevance to query
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                score = calculate_text_similarity(query, sentence)
                sentence_scores.append((score, i, sentence))
            
            # Sort by relevance and position (prefer earlier sentences)
            sentence_scores.sort(key=lambda x: (x[0], -x[1]), reverse=True)
            
            # Select top sentences that fit in token limit
            selected = []
            total_tokens = 0
            
            for score, idx, sentence in sentence_scores:
                tokens = estimate_tokens(sentence)
                if total_tokens + tokens <= max_tokens:
                    selected.append((idx, sentence))
                    total_tokens += tokens
                else:
                    break
            
            # Sort by original order
            selected.sort(key=lambda x: x[0])
            return " ".join([sent for _, sent in selected])
        
        else:
            # Simple truncation from beginning
            selected = []
            total_tokens = 0
            
            for sentence in sentences:
                tokens = estimate_tokens(sentence)
                if total_tokens + tokens <= max_tokens:
                    selected.append(sentence)
                    total_tokens += tokens
                else:
                    break
            
            return " ".join(selected)
    
    def _find_best_section(self, statement: str, articles: List[Dict[str, Any]]) -> str:
        """Find the most relevant section across all articles."""
        best_section = ""
        best_similarity = 0.0
        
        for article in articles:
            sections = self._split_into_sections(article['content'])
            
            for section in sections:
                if len(section.split()) > 20:  # Minimum section size
                    similarity = calculate_text_similarity(statement, section)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_section = section
        
        return best_section if best_similarity >= self.semantic_threshold else ""
    
    def _split_into_sections(self, content: str) -> List[str]:
        """Split content into meaningful sections."""
        sections = []
        current_section = []
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Check if this is a heading
            if re.match(r'^#+\s', line):
                # Save current section
                if current_section:
                    section_text = '\n'.join(current_section).strip()
                    if section_text:
                        sections.append(section_text)
                
                # Start new section
                current_section = [line]
            else:
                if line:
                    current_section.append(line)
        
        # Add final section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if section_text:
                sections.append(section_text)
        
        return sections
    
    def generate_dataset(self) -> List[Dict[str, Any]]:
        """Generate the improved dataset."""
        examples = []
        similarity_stats = []
        
        logger.info("Generating improved dataset...")
        
        for i, (statement, topic_id) in enumerate(zip(self.statements, self.statement_topics)):
            if topic_id not in self.documents_by_topic:
                continue
            
            topic_data = self.documents_by_topic[topic_id]
            
            # Select best positive context
            positive_context, similarity = self._select_best_context(statement, topic_data)
            
            if not positive_context or similarity < self.semantic_threshold:
                logger.debug(f"Skipped statement {i} (similarity: {similarity:.3f})")
                continue
            
            # Generate negatives
            negatives = self._generate_negatives(statement, topic_id)
            
            if negatives:
                examples.append({
                    'anchor': statement,
                    'positive': positive_context,
                    'negatives': negatives,
                    'similarity_score': similarity
                })
                similarity_stats.append(similarity)
            
            # Log progress
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(self.statements)} statements")
        
        # Log statistics
        if similarity_stats:
            avg_sim = sum(similarity_stats) / len(similarity_stats)
            min_sim = min(similarity_stats)
            max_sim = max(similarity_stats)
            
            logger.info(f"Generated {len(examples)} examples")
            logger.info(f"Similarity - Avg: {avg_sim:.3f}, Min: {min_sim:.3f}, Max: {max_sim:.3f}")
        
        return examples
    
    def _generate_negatives(self, statement: str, positive_topic_id: int) -> List[str]:
        """Generate high-quality negative examples."""
        negatives = []
        
        # Get candidate topics (exclude positive topic)
        candidate_topics = [tid for tid in self.documents_by_topic.keys() 
                          if tid != positive_topic_id]
        
        # Select random topics for negatives
        selected_topics = random.sample(
            candidate_topics, 
            min(self.negative_ratio * 2, len(candidate_topics))
        )
        
        for topic_id in selected_topics:
            if len(negatives) >= self.negative_ratio:
                break
            
            topic_data = self.documents_by_topic[topic_id]
            if topic_data['articles']:
                # Select first article as negative
                article = topic_data['articles'][0]
                content = article['content']
                
                # Truncate if needed
                if estimate_tokens(content) > self.max_tokens_per_example:
                    content = self._smart_truncate(content, max_tokens=self.max_tokens_per_example)
                
                negatives.append(content)
        
        return negatives
    
    def save_dataset(self, examples: List[Dict[str, Any]], output_path: str):
        """Save the improved dataset."""
        # Calculate statistics
        similarities = [ex['similarity_score'] for ex in examples]
        token_stats = {
            'avg_anchor_tokens': sum(estimate_tokens(ex['anchor']) for ex in examples) / len(examples),
            'avg_positive_tokens': sum(estimate_tokens(ex['positive']) for ex in examples) / len(examples),
            'avg_negative_tokens': sum(
                sum(estimate_tokens(neg) for neg in ex['negatives']) / len(ex['negatives'])
                for ex in examples
            ) / len(examples),
            'avg_similarity': sum(similarities) / len(similarities),
            'min_similarity': min(similarities),
            'max_similarity': max(similarities),
            'max_tokens_limit': self.max_tokens_per_example
        }
        
        # Remove similarity scores from examples (not needed for training)
        clean_examples = []
        for ex in examples:
            clean_examples.append({
                'anchor': ex['anchor'],
                'positive': ex['positive'],
                'negatives': ex['negatives']
            })
        
        data = {
            'examples': clean_examples,
            'generation_config': {
                'max_tokens_per_example': self.max_tokens_per_example,
                'use_full_articles': self.use_full_articles,
                'semantic_threshold': self.semantic_threshold,
                'negative_ratio': self.negative_ratio,
                'generation_timestamp': datetime.now().isoformat(),
                'improvements': [
                    'Semantic similarity-based context selection',
                    'Full article context when possible',
                    'Smart truncation preserving relevance',
                    'Medical term weighted similarity',
                    'Increased token limits'
                ]
            },
            'token_stats': token_stats,
            'dataset_stats': {
                'total_examples': len(clean_examples),
                'total_statements': len(self.statements),
                'total_topics': len(self.documents_by_topic),
                'average_similarity': token_stats['avg_similarity']
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Quality improvements:")
        for improvement in data['generation_config']['improvements']:
            logger.info(f"  + {improvement}")

def main():
    """Generate improved medical dataset."""
    print("Generating Improved Medical Embedding Dataset")
    print("=" * 60)
    
    try:
        # Initialize generator
        generator = ImprovedMedicalDatasetGenerator(
            max_tokens_per_example=512,    # Increased from ~200
            use_full_articles=True,        # Use complete articles when possible
            semantic_threshold=0.1,        # Minimum similarity for inclusion
            negative_ratio=3,              # Number of negatives per positive
            max_articles_per_topic=3       # Max articles to combine per topic
        )
        
        # Generate improved dataset
        examples = generator.generate_dataset()
        
        if not examples:
            logger.error("No examples generated! Check data paths and semantic threshold.")
            return
        
        # Save dataset
        output_path = "improved_medical_embedding_dataset.json"
        generator.save_dataset(examples, output_path)
        
        print(f"\nSUCCESS!")
        print(f"Generated improved dataset with {len(examples)} examples")
        print(f"Saved to: {output_path}")
        print("\nKey improvements over original:")
        print("  + Semantic similarity-based context selection")
        print("  + Full article context (512 tokens vs ~200)")
        print("  + Medical term weighted similarity scoring")  
        print("  + Smart truncation preserving relevance")
        print("  + Higher quality positive examples")
        
    except Exception as e:
        logger.error(f"Failed to generate dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()