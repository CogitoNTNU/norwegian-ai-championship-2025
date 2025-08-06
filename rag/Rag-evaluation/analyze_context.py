
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from templates.biobert_diagnostic import BiobertDiagnostic

# The question from Case 1
question = "The radial approach for coronary angiography carries a higher risk of complications compared to the femoral or brachial routes."

print("Initializing diagnostic system...")
diagnostic = BiobertDiagnostic()

print("\nRetrieving documents for the question...")
# Use the same method the main 'run' function now uses
retrieved_docs = diagnostic.topic_classified_bm25_retrieval(question, k=5)

print("\n--- Full Retrieved Context ---")
for i, result in enumerate(retrieved_docs):
    doc = result['document']
    text = doc.get('text', 'N/A')
    topic = doc.get('topic_name', 'Unknown')
    print(f"--- Document {i+1} (Topic: {topic}) ---")
    print(text)
    print("-" * 20)

print("\n--- Analysis ---")
print("The statement claims the 'radial approach' has a 'higher risk'.")
print("To contradict this, the context needs to state that the radial approach has a 'lower risk' or 'fewer complications' than the femoral/brachial routes.")
print("As you can see from the retrieved text, this specific comparison is not present.")

