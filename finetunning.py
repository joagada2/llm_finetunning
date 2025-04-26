#!/usr/bin/env python3
"""
finetune_stablelm_sst2.py

Fine-tunes stabilityai/stablelm-base-alpha-7b on SST-2 sentiment classification.
Assumes you have:
  finetune_data/
    ├ model/            ← cloned stablelm-base-alpha-7b repo (config.json, pytorch weights, tokenizer)
    ├ train.jsonl
    └ validation.jsonl
"""

import json
import numpy as np
from pathlib import Path

import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ——— Config ——————————————————————————————————————————
set_seed(42)

BASE_DIR        = Path(__file__).parent / "finetune_data"
MODEL_DIR       = BASE_DIR / "model"
TRAIN_JSONL     = BASE_DIR / "train.jsonl"
VALID_JSONL     = BASE_DIR / "validation.jsonl"
MAX_SEQ_LENGTH  = 128

OUTPUT_DIR      = "stablelm_finetuned"
LOG_DIR         = "stablelm_logs"

# ——— Load tokenizer & model —————————————————————————————
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=2,
    ignore_mismatched_sizes=True,  # adds a fresh head
)

# ——— Prepare dataset —————————————————————————————————————
raw_ds = load_dataset(
    "json",
    data_files={
        "train": str(TRAIN_JSONL),
        "validation": str(VALID_JSONL),
    }
)

def preprocess_fn(batch):
    toks = tokenizer(
        batch["sentence"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )
    toks["labels"] = batch["label"]
    return toks

tokenized = raw_ds.map(
    preprocess_fn,
    batched=True,
    remove_columns=raw_ds["train"].column_names,
)

tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ——— Metrics ————————————————————————————————————————
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# ——— TrainingArguments ——————————————————————————————
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=LOG_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    seed=42,
)

# ——— Trainer ————————————————————————————————————————
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ——— Train ————————————————————————————————————————
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/best")
    print(f"✅ Fine-tuning complete. Best model saved to {OUTPUT_DIR}/best")
