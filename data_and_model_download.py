#!/usr/bin/env python3
"""
prepare_finetune.py

Downloads QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF model
and the GLUE SST-2 sentiment dataset, tokenizes/preprocesses, and
saves everything locally for downstream fine-tuning.
"""

import os
from pathlib import Path

from huggingface_hub import snapshot_download
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# 1. Configuration
MODEL_ID = "QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF"
DATASET_NAME = "glue"
DATASET_CONFIG = "sst2"
MAX_LENGTH = 128
OUTPUT_DIR = Path("finetune_data")
MODEL_DIR = OUTPUT_DIR / "model"
TOKENIZER_DIR = OUTPUT_DIR / "tokenizer"
DATA_DIR = OUTPUT_DIR / "dataset"

# Make output directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 2. Download model files via huggingface_hub.snapshot_download
print(f">>> Downloading model repo `{MODEL_ID}` …")
repo_local_path = snapshot_download(repo_id=MODEL_ID)
print(f">>> Model repo cached at {repo_local_path}")

# 3. Load tokenizer & model for sequence classification (2 labels: pos/neg)
print(">>> Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(repo_local_path, use_fast=True)
print(">>> Loading model with classification head …")
model = AutoModelForSequenceClassification.from_pretrained(
    repo_local_path,
    num_labels=2,                # SST-2 is binary
    ignore_mismatched_sizes=True # allow adding head on top of a LM
)

# 4. Save tokenizer & model locally for trainer to pick up
print(f">>> Saving tokenizer to {TOKENIZER_DIR}")
tokenizer.save_pretrained(TOKENIZER_DIR)
print(f">>> Saving model to {MODEL_DIR}")
model.save_pretrained(MODEL_DIR)

# 5. Load the SST-2 dataset
print(f">>> Loading dataset {DATASET_NAME}/{DATASET_CONFIG} …")
raw_datasets = load_dataset(DATASET_NAME, DATASET_CONFIG)

# 6. Tokenization / preprocessing function
def preprocess_fn(examples):
    # GLUE/SST2 uses 'sentence' field
    texts = examples["sentence"]
    # pad & truncate
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

# 7. Apply preprocessing
print(">>> Tokenizing dataset …")
tokenized = raw_datasets.map(
    preprocess_fn,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

# 8. Save tokenized dataset to disk
print(f">>> Saving tokenized dataset to {DATA_DIR}")
tokenized.save_to_disk(DATA_DIR)

print("✅ All set! You can now launch your Trainer on:")
print(f"   · model:     {MODEL_DIR}")
print(f"   · tokenizer: {TOKENIZER_DIR}")
print(f"   · dataset:   {DATA_DIR}")
