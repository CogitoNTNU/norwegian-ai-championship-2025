#!/usr/bin/env python3
"""
Integration script to connect the RAG system with the competition framework.
This updates the original model.py to use our RAG classifier.
"""

import os
import shutil
from pathlib import Path

def backup_original_model():
    """Backup the original model.py file."""
    project_root = Path(__file__).parent.parent.parent
    competition_dir = project_root / "DM-i-AI-2025" / "emergency-healthcare-rag"
    
    original_model = competition_dir / "model.py"
    backup_model = competition_dir / "model_original.py"
    
    if original_model.exists() and not backup_model.exists():
        shutil.copy2(original_model, backup_model)
        print("✅ Original model.py backed up as model_original.py")
    else:
        print("ℹ️  Original model.py already backed up or doesn't exist")

def create_new_model_py():
    """Create new model.py that uses our RAG classifier."""
    project_root = Path(__file__).parent.parent.parent
    competition_dir = project_root / "DM-i-AI-2025" / "emergency-healthcare-rag"
    
    new_model_content = '''import sys
import os
from pathlib import Path
from typing import Tuple

# Add our RAG system to the path
current_dir = Path(__file__).parent
rag_dir = current_dir.parent.parent / "src" / "rag"
sys.path.insert(0, str(rag_dir))

try:
    from classifier import predict as rag_predict
    RAG_AVAILABLE = True
    print("🤖 RAG classifier loaded successfully")
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"⚠️  RAG classifier not available: {e}")
    print("Falling back to baseline model")

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
    if RAG_AVAILABLE:
        try:
            return rag_predict(statement)
        except Exception as e:
            print(f"Error in RAG prediction: {e}")
            print("Falling back to baseline")
    
    # Fallback to baseline implementation
    return baseline_predict(statement)

def baseline_predict(statement: str) -> Tuple[int, int]:
    """Baseline prediction (same as original)."""
    import json
    
    # Naive baseline that always returns True for statement classification
    statement_is_true = 1

    # Simple topic matching based on keywords in topic names
    statement_topic = match_topic(statement)

    return statement_is_true, statement_topic

def match_topic(statement: str) -> int:
    """Simple keyword matching to find the best topic match."""
    # Load topics mapping
    try:
        with open("data/topics.json", "r") as f:
            topics = json.load(f)
    except FileNotFoundError:
        return 0

    statement_lower = statement.lower()
    best_topic = 0
    max_matches = 0

    for topic_name, topic_id in topics.items():
        # Extract keywords from topic name
        keywords = (
            topic_name.lower()
            .replace("_", " ")
            .replace("(", "")
            .replace(")", "")
            .split()
        )

        # Count keyword matches in statement
        matches = sum(1 for keyword in keywords if keyword in statement_lower)

        if matches > max_matches:
            max_matches = matches
            best_topic = topic_id

    return best_topic
'''
    
    new_model_file = competition_dir / "model.py"
    with open(new_model_file, 'w') as f:
        f.write(new_model_content)
    
    print("✅ New model.py created with RAG integration")

def test_integration():
    """Test that the integration works."""
    project_root = Path(__file__).parent.parent.parent
    competition_dir = project_root / "DM-i-AI-2025" / "emergency-healthcare-rag"
    
    # Change to competition directory for testing
    original_cwd = os.getcwd()
    os.chdir(competition_dir)
    
    try:
        # Import and test the new model
        import sys
        sys.path.insert(0, str(competition_dir))
        
        from model import predict
        
        test_statement = "Testicular torsion requires immediate surgical intervention to prevent testicular loss."
        result = predict(test_statement)
        
        print(f"✅ Integration test successful!")
        print(f"   Statement: {test_statement}")
        print(f"   Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def main():
    """Main integration function."""
    print("🔗 Integrating RAG system with competition framework")
    print("=" * 55)
    
    # Step 1: Backup original model
    print("\n1. Backing up original model...")
    backup_original_model()
    
    # Step 2: Create new model.py
    print("\n2. Creating new model.py...")
    create_new_model_py()
    
    # Step 3: Test integration
    print("\n3. Testing integration...")
    success = test_integration()
    
    print("\n" + "=" * 55)
    if success:
        print("🎉 Integration completed successfully!")
        print("\nYour competition model.py now uses the RAG system!")
        print("\nTo test the competition framework:")
        print("1. cd DM-i-AI-2025/emergency-healthcare-rag")
        print("2. python example.py")
        print("3. python api.py")
    else:
        print("❌ Integration failed. Check the errors above.")
    
    return success

if __name__ == "__main__":
    main()
