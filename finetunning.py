#!/usr/bin/env python3
import json
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

# ——— Paths —————————————————————————
BASE        = Path(__file__).parent / "finetune_data"
MODEL_DIR   = BASE / "model"
TOKENIZER_DIR = MODEL_DIR      # tokenizer lives alongside
TRAIN_FILE  = BASE / "train.jsonl"
VAL_FILE    = BASE / "validation.jsonl"

# ——— Load model & tokenizer —————————————————
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=2,
    ignore_mismatched_sizes=True,
)

# ——— Load raw JSONL, tokenize ——————————————
raw = load_dataset("json", data_files={
    "train": str(TRAIN_FILE),
    "validation": str(VAL_FILE),
})

def prep(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

tokenized = raw.map(
    prep, batched=True,
    remove_columns=raw["train"].column_names
)
tokenized.set_format("torch", columns=["input_ids","attention_mask","label"])

# ——— Metrics —————————————————————————————
acc = evaluate.load("accuracy")
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    return acc.compute(predictions=preds, references=p.label_ids)

# ——— Training setup ——————————————————————
args = TrainingArguments(
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
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("finetuned_model/best")
    print("🎉 Fine-tuning complete!")
