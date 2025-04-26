#!/usr/bin/env python3
"""
finetune_stablelm_sst2.py

Fine-tunes stabilityai/stablelm-base-alpha-7b on SST-2 sentiment classification,
while redirecting all HF caches to ./hf_cache to avoid home-dir quotas.
"""

import os
import json
import numpy as np
from pathlib import Path

# ── 0) Redirect all HF caches to a local hf_cache folder ────────────────────
BASE_PATH = Path(__file__).parent
HF_CACHE  = BASE_PATH / "hf_cache"
for sub in ("transformers","datasets","metrics","hub"):
    (HF_CACHE / sub).mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"]            = str(HF_CACHE / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")
os.environ["HF_DATASETS_CACHE"]  = str(HF_CACHE / "datasets")
os.environ["HF_METRICS_CACHE"]   = str(HF_CACHE / "metrics")
os.environ["HF_HUB_CACHE"]       = str(HF_CACHE / "hub")

# ── 1) Imports (after setting env) ─────────────────────────────────────────
import evaluate
from datasets import load_dataset
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ── 2) Configuration ───────────────────────────────────────────────────────
set_seed(42)
BASE_DIR       = BASE_PATH / "finetune_data"
MODEL_DIR      = BASE_DIR / "model"
TRAIN_JSONL    = BASE_DIR / "train.jsonl"
VALID_JSONL    = BASE_DIR / "validation.jsonl"
MAX_SEQ_LENGTH = 128

OUTPUT_DIR     = "stablelm_finetuned"
LOG_DIR        = "stablelm_logs"

# ── 3) Load tokenizer & model ——————————————————————————————————————
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
# Tell the tokenizer to pad with the EOS token:
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=2,
    ignore_mismatched_sizes=True,
)
# Resize embeddings in case new tokens were added:
model.resize_token_embeddings(len(tokenizer))

# ── 4) Prepare datasets ————————————————————————————————————————
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

# ── 5) Metrics ——————————————————————————————————————————————
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# ── 6) TrainingArguments —————————————————————————————————————
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

# ── 7) Trainer ——————————————————————————————————————————————
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ── 8) Train ——————————————————————————————————————————————
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/best")
    print(f"✅ Fine-tuning complete. Best model saved to {OUTPUT_DIR}/best")
