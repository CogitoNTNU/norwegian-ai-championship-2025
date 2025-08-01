
import json
from typing import Tuple, Dict

def validate_prediction(statement_is_true: int, statement_topic: int):
    """Validate that prediction values are in correct format"""
    assert isinstance(statement_is_true, int), f"statement_is_true must be int, got {type(statement_is_true)}"
    assert isinstance(statement_topic, int), f"statement_topic must be int, got {type(statement_topic)}"
    assert statement_is_true in [0, 1], f"statement_is_true must be 0 or 1, got {statement_is_true}"
    assert 0 <= statement_topic <= 114, f"statement_topic must be between 0-114, got {statement_topic}"

def load_statement_sample(statement_id: str) -> Tuple[str, Dict]:
    """Load a statement and its answer from training data"""
    statement_file = f"data/train/statements/statement_{statement_id}.txt"
    answer_file = f"data/train/answers/statement_{statement_id}.json"
    
    with open(statement_file, 'r') as f:
        statement = f.read().strip()
    
    with open(answer_file, 'r') as f:
        answer = json.load(f)
    
    return statement, answer
