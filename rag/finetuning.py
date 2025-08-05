from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
import os
import json
from typing import List, Dict


# ========================================
# 1. Load Model
# ========================================
model, tokenizer = FastLanguageModel.from_pretrained(
    "cogitoai/cogito-8b",
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)


# ========================================
# 2. Load Topics (by folder name = topic ID)
# ========================================
def load_topics_as_dict(topics_dir: str) -> Dict[int, str]:
    """
    Returns a dict: topic_id (int) -> combined markdown content
    Assumes folder names are integers (e.g., '0', '1', ..., '114')
    """
    topic_contents = {}
    for folder_name in os.listdir(topics_dir):
        folder_path = os.path.join(topics_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        try:
            topic_id = int(folder_name)
        except ValueError:
            print(f"Skipping non-topic folder: {folder_name}")
            continue

        combined = []
        for file in os.listdir(folder_path):
            if file.endswith(".md"):
                file_path = os.path.join(folder_path, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        combined.append(content)
        topic_contents[topic_id] = (
            "\n\n".join(combined) if combined else "No content available."
        )
    return topic_contents


topics_dir = "data/raw/topics"
topic_id_to_context = load_topics_as_dict(topics_dir)
print(f"Loaded {len(topic_id_to_context)} topics.")


# ========================================
# 3. Load Statements & Answers and Match Them
# ========================================
def load_statements_and_answers(
    statement_dir: str, answer_dir: str
) -> List[Dict[str, str]]:
    """
    Returns list of dicts: {
        "statement": "High BP causes stroke...",
        "context": "...",
        "output": '{"statement_is_true": 1, "statement_topic": 42}'
    }
    Assumes filenames match: e.g., stmt_001.txt ↔ ans_001.json
    """
    examples = []

    # Get sorted lists to ensure alignment
    statement_files = sorted(
        [f for f in os.listdir(statement_dir) if f.endswith(".txt")]
    )
    answer_files = sorted([f for f in os.listdir(answer_dir) if f.endswith(".json")])

    if len(statement_files) != len(answer_files):
        raise ValueError("Mismatch in number of statements and answers!")

    for s_file, a_file in zip(statement_files, answer_files):
        # Extract ID for alignment check
        stmt_id = os.path.splitext(s_file.split("_")[-1])[
            0
        ]  # Handles 'stmt_001.txt' → '001'
        ans_id = os.path.splitext(a_file.split("_")[-1])[0]  # Same for answer
        if stmt_id != ans_id:
            print(f"Warning: Mismatched IDs? {s_file} vs {a_file}")

        # ✅ Load statement from .txt as plain text
        with open(os.path.join(statement_dir, s_file), "r", encoding="utf-8") as f:
            statement_text = f.read().strip()

        # Load answer (JSON)
        with open(os.path.join(answer_dir, a_file), "r", encoding="utf-8") as f:
            ans_data = json.load(f)
        is_true = ans_data.get("statement_is_true", 0)
        topic_id = ans_data.get("statement_topic", 0)

        # Format expected output as stringified JSON
        label_str = f'{{"statement_is_true": {is_true}, "statement_topic": {topic_id}}}'

        # Retrieve context
        context = topic_id_to_context.get(topic_id, "No relevant context found.")

        examples.append(
            {"statement": statement_text, "context": context, "output": label_str}
        )

    return examples


# Load data
train_data = load_statements_and_answers(
    statement_dir="data/raw/train/statements", answer_dir="data/raw/train/answers"
)

print(f"Loaded {len(train_data)} aligned (statement, answer, context) examples.")


# ========================================
# 4. Create Dataset
# ========================================
example = {"statement_is_true": 1, "statement_topic": 42}
instruction = (
    "Evaluate the truthfulness of the medical statement based on the provided context.\n"
    "Respond ONLY with a JSON object like this:\n"
    f"{json.dumps(example)}\n"
    "Where:\n"
    "  - statement_is_true: 1 if true, 0 if false\n"
    "  - statement_topic: integer from 0 to 114 representing the relevant medical topic\n"
    "Do not include any other text or formatting."
)

dataset = Dataset.from_dict(
    {
        "instruction": [instruction] * len(train_data),
        "context": [ex["context"] for ex in train_data],
        "statement": [ex["statement"] for ex in train_data],
        "output": [ex["output"] for ex in train_data],
    }
)


# ========================================
# 5. Format for Training
# ========================================
def formatting_prompts_func(examples):
    texts = []
    for inst, ctx, stmt, out in zip(
        examples["instruction"],
        examples["context"],
        examples["statement"],
        examples["output"],
    ):
        text = (
            f"### Instruction\n{inst.strip()}\n\n"
            f"### Context\n{ctx.strip()}\n\n"
            f"### Statement\n{stmt.strip()}\n\n"
            f"### Response\n{out.strip()}"
        )
        texts.append(text)
    return {"text": texts}


dataset = dataset.map(formatting_prompts_func, batched=True)


# ========================================
# 6. Train
# ========================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_8bit",
        seed=42,
        save_strategy="epoch",
    ),
)

trainer.train()


# ========================================
# 7. Save
# ========================================
model.save_pretrained("fine_tuned_cogito_medical")
model.save_pretrained_merged(
    "cogito-medical-merged", tokenizer, save_method="merged_16bit"
)

# ========================================
# To get started with the fine-tuned model:
# ========================================
# from unsloth import FastLanguageModel

# model, tokenizer = FastLanguageModel.from_pretrained(
#     "cogito-medical-merged",  # ← Finetuned model
#     load_in_4bit=True,
# )
