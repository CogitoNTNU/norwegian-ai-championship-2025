from utils import load_statement_sample
from model import predict

# Load a sample from training data
statement, true_answer = load_statement_sample("0001")

print(f"Statement: {statement}")
print(f"True answer: {true_answer}")

# Make prediction
statement_is_true, statement_topic = predict(statement)

print(
    f"Predicted: statement_is_true={statement_is_true}, statement_topic={statement_topic}"
)

# Check accuracy
true_correct = statement_is_true == true_answer["statement_is_true"]
topic_correct = statement_topic == true_answer["statement_topic"]

print(f"Truth prediction correct: {true_correct}")
print(f"Topic prediction correct: {topic_correct}")
print(f"Both correct: {true_correct and topic_correct}")
