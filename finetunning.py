#!/usr/bin/env python3
"""
finetune_stablelm_sst2_no_eval_dep.py

Fine-tunes stabilityai/stablelm-base-alpha-7b on SST-2 sentiment classification,
redirecting all HF caches to ./hf_cache to avoid home-dir quotas, and
computes accuracy manually (no `evaluate` lib).
"""

import os
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score

# ── 0) Redirect HF caches to a local folder ──────────────────────────────────
BASE_PATH = Path(__file__).parent
HF_CACHE  = BASE_PATH / "hf_cache"
for sub in ("transformers","datasets","metrics","hub"):
    (HF_CACHE / sub).mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"]            = str(HF_CACHE / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")
os.environ["HF_DATASETS_CACHE"]  = str(HF_CACHE / "datasets")
os.environ["HF_METRICS_CACHE"]   = str(HF_CACHE / "metrics")
os.environ["HF_HUB_CACHE"]       = str(HF_CACHE / "hub")

# ── Imports (after cache setup) ─────────────────────────────────────────────
from datasets import load_dataset
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ── Config ───────────────────────────────────────────────────────────────────
set_seed(42)
BASE_DIR       = BASE_PATH / "finetune_data"
MODEL_DIR      = BASE_DIR / "model"
TRAIN_JSONL    = BASE_DIR / "train.jsonl"
VALID_JSONL    = BASE_DIR / "validation.jsonl"
MAX_SEQ_LENGTH = 128

OUTPUT_DIR     = "stablelm_finetuned"
LOG_DIR        = "stablelm_logs"

# ── Load tokenizer & model ───────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=2,
    ignore_mismatched_sizes=True,
)
model.resize_token_embeddings(len(tokenizer))

# ── Prepare datasets ─────────────────────────────────────────────────────────
raw_ds = load_dataset(
    "json",
    data_files={"train": str(TRAIN_JSONL), "validation": str(VALID_JSONL)},
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
tokenized.set_format("torch", columns=["input_ids","attention_mask","labels"])

# ── Compute metrics manually ─────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# ── TrainingArguments ────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir=LOG_DIR,
    logging_steps=50,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    tokenizer=tokenizer,
)

if __name__ == "__main__":
    trainer.train()
    # Manually evaluate:
    metrics = trainer.evaluate(eval_dataset=tokenized["validation"])
    print("Validation metrics:", metrics)
    trainer.save_model(f"{OUTPUT_DIR}/final")
