#!/usr/bin/env python3
import os
from huggingface_hub import Repository
from datasets import load_dataset

OUT_DIR     = "finetune_data"
MODEL_ID    = "QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF"
DATA_NAME   = "glue"
DATA_CONFIG = "sst2"

os.makedirs(OUT_DIR, exist_ok=True)

# A) Clone full transformers repo
print("Cloning the transformer-compatible model repo …")
Repository(local_dir=os.path.join(OUT_DIR,"model"),
           clone_from=MODEL_ID)

# B) Dump raw SST-2 into JSONL
print("Downloading SST-2 splits …")
ds = load_dataset(DATA_NAME, DATA_CONFIG)
for split, data in ds.items():
    out_path = os.path.join(OUT_DIR, f"{split}.jsonl")
    print(f" • {split} → {out_path}")
    data.to_json(out_path, orient="records", lines=True)

print("✔️ All raw data + model repo ready in ./finetune_data/")
