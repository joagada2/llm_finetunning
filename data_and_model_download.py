#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download
from datasets import load_dataset

MODEL_ID    = "QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF"
DATA_NAME   = "glue"
DATA_CONFIG = "sst2"
OUT_DIR     = "finetune_data"

os.makedirs(OUT_DIR, exist_ok=True)

# 1) Download the HF repo (model files + tokenizer JSON)
print("Downloading model repo…")
snapshot_download(MODEL_ID, cache_dir=os.path.join(OUT_DIR, "model"))

# 2) Pull raw SST-2 splits, save as JSONL
print("Downloading SST-2 dataset…")
ds = load_dataset(DATA_NAME, DATA_CONFIG)
for split, data in ds.items():
    out_path = os.path.join(OUT_DIR, f"{split}.jsonl")
    print(f"  • Saving {split} → {out_path}")
    data.to_json(out_path, orient="records", lines=True)

print("✅ Finished. Model repo + raw data are in ./finetune_data/")
