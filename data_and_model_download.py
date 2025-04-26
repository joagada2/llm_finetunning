#!/usr/bin/env python3
"""
prepare_finetune.py

— Uses a user-writable hf_cache directory (env var HF_SCRATCH, or ./hf_cache)
— Downloads QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF for fine-tuning.
— Streams & tokenizes SST-2, writing out JSONL shards.
"""

import os
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1) Decide where to put all HF caches:
#
#    You can override by setting HF_SCRATCH in your shell:
#      export HF_SCRATCH=/path/to/your/writable/cache
#
#    Otherwise it will create "./hf_cache" next to this script.
scratch_env = os.environ.get("HF_SCRATCH")
if scratch_env:
    SCRATCH = Path(scratch_env)
else:
    SCRATCH = Path(__file__).resolve().parent / "hf_cache"

# create it if needed
SCRATCH.mkdir(parents=True, exist_ok=True)

# Tell HF libs to use it
for k in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "HF_METRICS_CACHE"):
    os.environ[k] = str(SCRATCH)

# 2) Config
MODEL_ID    = "QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF"
OUT_ROOT    = Path(__file__).resolve().parent / "finetune_data"
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
)
print(">>> Loading model (seq-class head) …")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,                 # SST-2: pos/neg
    ignore_mismatched_sizes=True,
)

# 4) Persist to project folder for Trainer
print(f">>> Saving tokenizer → {OUT_TOKEN}")
tokenizer.save_pretrained(OUT_TOKEN)
print(f">>> Saving model     → {OUT_MODEL}")
model.save_pretrained(OUT_MODEL)

# 5) Stream and tokenize SST-2, write out JSONL
print(">>> Streaming & tokenizing SST-2 …")
for split in ("train", "validation", "test"):
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

print("✅ Done!  Caches in", SCRATCH)
print("   Model+tokenizer in", OUT_MODEL)
print("   SST-2 shards in", OUT_ROOT)
