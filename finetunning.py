#!/usr/bin/env python3
"""
finetune.py

Fine-tunes QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF
on SST-2 sentiment classification.
"""

import numpy as np
from pathlib import Path

from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ——— 1) Paths ——————————————————————————————————————————————
BASE_DIR      = Path(__file__).resolve().parent / "finetune_data"
MODEL_DIR     = BASE_DIR / "model"
TOKENIZER_DIR = BASE_DIR / "model"   # tokenizer files live in the same repo clone
TRAIN_FILE    = BASE_DIR / "train.jsonl"
VAL_FILE      = BASE_DIR / "validation.jsonl"

# ——— 2) Load tokenizer & model ————————————————————————————
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=2,
    ignore_mismatched_sizes=True,  # adds a fresh head even if shape mismatches
)

# ——— 3) Load JSONL dataset splits ——————————————————————————
data_files = {
    "train": str(TRAIN_FILE),
    "validation": str(VAL_FILE),
}
raw_datasets = load_dataset("json", data_files=data_files)

# ——— 4) Preprocessing ————————————————————————————————————
def preprocess_batch(examples):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

tokenized = raw_datasets.map(
    preprocess_batch,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

# cast to PyTorch tensors
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ——— 5) Metrics ————————————————————————————————————————
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)

# ——— 6) Training Arguments —————————————————————————————
training_args = TrainingArguments(
    output_dir="finetuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="logs",
    logging_steps=50,
)

# ——— 7) Trainer ————————————————————————————————————————
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ——— 8) Train & Save ————————————————————————————————————
if __name__ == "__main__":
    trainer.train()
    trainer.save_model("finetuned_model/best")
    print("✅ Fine-tuning complete. Best model saved to finetuned_model/best")
