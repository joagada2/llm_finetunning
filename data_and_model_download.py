#!/usr/bin/env python3
"""
prepare_finetune.py

— Uses a scratch area for all HF caches to avoid home-dir quota issues.
— Streams & tokenizes SST-2 in one pass, writing out JSONL shards.
— Downloads QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF for fine-tuning.
"""

import os
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1) Point HF caches at a higher-quota mount (e.g. /scratch/$USER/hf)
USER = os.environ.get("USER", "")
SCRATCH = f"/scratch/{USER}/hf_cache"
for k in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "HF_METRICS_CACHE"):
    os.environ[k] = SCRATCH
Path(SCRATCH).mkdir(parents=True, exist_ok=True)

# 2) Config
MODEL_ID    = "QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF"
OUT_ROOT    = Path("finetune_data")
MAX_LENGTH  = 128
OUT_MODEL   = OUT_ROOT / "model"
OUT_TOKEN   = OUT_ROOT / "tokenizer"

OUT_ROOT.mkdir(exist_ok=True)
OUT_MODEL.mkdir(exist_ok=True)
OUT_TOKEN.mkdir(exist_ok=True)

# 3) Download & cache model + tokenizer
print(">>> Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
    cache_dir=SCRATCH,
)
print(">>> Loading model (seq-class head) …")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,                # SST-2: pos/neg
    ignore_mismatched_sizes=True, 
    cache_dir=SCRATCH,
)

# 4) Persist to project folder for Trainer
print(f">>> Saving tokenizer → {OUT_TOKEN}")
tokenizer.save_pretrained(OUT_TOKEN)
print(f">>> Saving model     → {OUT_MODEL}")
model.save_pretrained(OUT_MODEL)

# 5) Stream and tokenize SST-2, write out JSONL
print(">>> Streaming & tokenizing SST-2 …")
splits = ["train", "validation", "test"]
for split in splits:
    ds = load_dataset("glue", "sst2", split=split, streaming=True)
    out_file = OUT_ROOT / f"{split}.jsonl"
    print(f"    • {split}: writing → {out_file}")
    with open(out_file, "w") as fp:
        for ex in ds:
            toks = tokenizer(
                ex["sentence"],
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
            )
            record = {
                "input_ids": toks["input_ids"],
                "attention_mask": toks["attention_mask"],
                "label": ex["label"],
            }
            fp.write(json.dumps(record) + "\n")

print("✅ All done!  Model+tokenizer in finetune_data/, SST-2 shards as JSONL.") 
