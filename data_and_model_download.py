#!/usr/bin/env python3
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from datasets import load_dataset

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_ID    = "QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF"
DATA_NAME   = "glue"
DATA_CONFIG = "sst2"
OUT_DIR     = Path("finetune_data")

# ── SETUP ────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR = OUT_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ── 1) Download the HF model repo via HTTP (no git-lfs) ──────────────────────
print("Downloading model repo via snapshot_download…")
repo_path = snapshot_download(
    repo_id=MODEL_ID,
    cache_dir=str(MODEL_DIR),
    repo_type="model",       # default, but explicit here
    local_files_only=False   # ensure we hit the network if not cached
)
print(f"✓ Model files written to {MODEL_DIR}")

# ── 2) Pull raw SST-2 splits and save as JSONL ───────────────────────────────
print("Downloading SST-2 dataset…")
ds = load_dataset(DATA_NAME, DATA_CONFIG)
for split, data in ds.items():
    out_file = OUT_DIR / f"{split}.jsonl"
    print(f"  • Saving {split} → {out_file}")
    data.to_json(out_file, orient="records", lines=True)

print("✅ Finished. Repo + data are in ./finetune_data/")
