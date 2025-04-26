#!/usr/bin/env python3
"""
prepare_and_download_finetune.py

1) snapshot_download the HF model (no git-lfs)
2) load via Transformers and re-save a proper model+tokenizer in finetune_data/model/
3) load SST-2 & dump train/validation/test as JSONL into finetune_data/
"""

import os
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_ID="meta-llama/Llama-2-7b"
DATA_NAME   = "glue"
DATA_CONFIG = "sst2"
OUT_ROOT    = Path("finetune_data")
MODEL_DIR   = OUT_ROOT / "model"

# ── PREPARE FOLDERS ──────────────────────────────────────────────────────────
OUT_ROOT.mkdir(exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── 1) Download the HF repo via HTTP ─────────────────────────────────────────
print("1) snapshot_download (HTTP) → fetching all repo files…")
cache_path = snapshot_download(
    repo_id=MODEL_ID,
    repo_type="model",
    local_files_only=False
)
print("   → cached at:", cache_path)

# ── 2) Load + re-export a real Transformers repo ──────────────────────────────
print("2) Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(cache_path, use_fast=True)

print("3) Loading base LM model (for config)…")
base_lm = AutoModelForCausalLM.from_pretrained(cache_path)

print("4) Loading sequence-classification head…")
cls_model = AutoModelForSequenceClassification.from_pretrained(
    cache_path,
    num_labels=2,
    ignore_mismatched_sizes=True,
)

print(f"5) Saving model+tokenizer → {MODEL_DIR}")
tokenizer.save_pretrained(MODEL_DIR)
cls_model.save_pretrained(MODEL_DIR)

# ── 3) Download SST-2 & dump JSONL ────────────────────────────────────────────
print("6) Downloading SST-2 splits and writing JSONL…")
ds = load_dataset(DATA_NAME, DATA_CONFIG)
for split, data in ds.items():
    out_file = OUT_ROOT / f"{split}.jsonl"
    print(f"   • {split} → {out_file}")
    # write newline-delimited JSON
    with open(out_file, "w") as fp:
        for record in data:
            fp.write(json.dumps({
                "sentence":    record["sentence"],
                "label":       record["label"],
            }) + "\n")

print("\n✅ All set! You now have:")
print("   • Model + tokenizer in:", MODEL_DIR)
print("   • SST-2 JSONL splits in:", OUT_ROOT) 
