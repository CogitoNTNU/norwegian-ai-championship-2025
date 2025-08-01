import json
from typing import Tuple

### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###
def predict(statement: str) -> Tuple[int, int]:
    """
    Predict both binary classification (true/false) and topic classification for a medical statement.
    
    Args:
        statement (str): The medical statement to classify
        
    Returns:
        Tuple[int, int]: (statement_is_true, statement_topic)
            - statement_is_true: 1 if true, 0 if false
            - statement_topic: topic ID from 0-114
    """
    # Naive baseline that always returns True for statement classification
    statement_is_true = 1
    
    # Simple topic matching based on keywords in topic names
    statement_topic = match_topic(statement)
    
    return statement_is_true, statement_topic

def match_topic(statement: str) -> int:
    """
    Simple keyword matching to find the best topic match.
    """
    # Load topics mapping
    with open('data/topics.json', 'r') as f:
        topics = json.load(f)
    
    statement_lower = statement.lower()
    best_topic = 0
    max_matches = 0
    
    for topic_name, topic_id in topics.items():
        # Extract keywords from topic name
        keywords = topic_name.lower().replace('_', ' ').replace('(', '').replace(')', '').split()
        
        # Count keyword matches in statement
        matches = sum(1 for keyword in keywords if keyword in statement_lower)
        
        if matches > max_matches:
            max_matches = matches
            best_topic = topic_id
    
    return best_topic
