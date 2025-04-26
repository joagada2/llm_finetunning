#!/usr/bin/env python3
"""
gpu_finetune_legacy.py

Fine-tunes stabilityai/stablelm-base-alpha-7b on SST-2 with LoRA,
DeepSpeed Stage2 offload, mixed-precision, and legacy TrainingArguments
(no evaluation_strategy).
"""

import os, json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from transformers import (
    set_seed, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
# … rest of your imports …


# pick a guaranteed‐writable cache location (beside this script)
BASE_DIR = Path(__file__).parent
HF_CACHE = BASE_DIR / "hf_cache"

# create subfolders
for sub in ("transformers","datasets","metrics","hub"):
    (HF_CACHE / sub).mkdir(parents=True, exist_ok=True)

# redirect ALL HF caches
os.environ["HF_HOME"]             = str(HF_CACHE / "hub")
os.environ["TRANSFORMERS_CACHE"]  = str(HF_CACHE / "transformers")
os.environ["HF_DATASETS_CACHE"]   = str(HF_CACHE / "datasets")
os.environ["HF_METRICS_CACHE"]    = str(HF_CACHE / "metrics")
os.environ["HF_HUB_CACHE"]        = str(HF_CACHE / "hub")

# ── 0) HF cache + disable wandb ──────────────────────────────────────────────
BASE = Path(__file__).parent
hf_cache = BASE/"hf_cache"
for sub in ("transformers","datasets","metrics","hub"):
    (hf_cache/sub).mkdir(parents=True, exist_ok=True)
os.environ.update({
    "HF_HOME": str(hf_cache/"hub"),
    "TRANSFORMERS_CACHE": str(hf_cache/"transformers"),
    "HF_DATASETS_CACHE": str(hf_cache/"datasets"),
    "HF_METRICS_CACHE": str(hf_cache/"metrics"),
    "HF_HUB_CACHE": str(hf_cache/"hub"),
    "WANDB_DISABLED": "true",
})

# ── 1) Config ────────────────────────────────────────────────────────────────
set_seed(42)
data_dir    = BASE/"finetune_data"
model_dir   = data_dir/"model"
train_file  = data_dir/"train.jsonl"
valid_file  = data_dir/"validation.jsonl"
MAX_LEN     = 128

# DeepSpeed Stage2 offload
ds_conf = {
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device":"cpu"},
    "offload_param":     {"device":"cpu"}
  },
  "train_batch_size": 16,
  "gradient_accumulation_steps": 2,
  "fp16": {"enabled": True}
}
with open("ds_config.json","w") as f:
    json.dump(ds_conf, f)

# ── 2) Model & Tokenizer ─────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    model_dir, num_labels=2, ignore_mismatched_sizes=True
)
model.config.pad_token_id = tokenizer.eos_token_id
model.resize_token_embeddings(len(tokenizer))

# LoRA on NeoX modules
lora_cfg = LoraConfig(r=16, lora_alpha=32,
    target_modules=["query_key_value","dense"])
model = get_peft_model(model, lora_cfg)

# ── 3) Dataset Prep ─────────────────────────────────────────────────────────
raw = load_dataset(
    "json",
    data_files={"train": str(TRAIN_FILE), "validation": str(VALID_FILE)}
)

def prep(batch):
    toks = tokenizer(batch["sentence"],
        truncation=True, padding="max_length", max_length=MAX_LEN)
    toks["labels"] = batch["label"]
    return toks

tokenized = raw.map(prep, batched=True,
    num_proc=os.cpu_count(),
    remove_columns=raw["train"].column_names)
tokenized.set_format("torch",
    columns=["input_ids","attention_mask","labels"])

# ── 4) Metrics fn ────────────────────────────────────────────────────────────
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# ── 5) TrainingArguments (legacy) ────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="gpu_finetuned_legacy",
    logging_dir="logs_legacy",
    num_train_epochs=3,
    per_device_train_batch_size=4,     # per-GPU
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    gradient_checkpointing=True,
    deepspeed="ds_config.json",
    logging_steps=50,
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    # no evaluation_strategy, no save_strategy, no load_best_model…
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    # 6) Train
    trainer.train()
    # 7) Manual evaluation
    print(">> Running final evaluation …")
    metrics = trainer.evaluate(eval_dataset=tokenized["validation"])
    print("Validation metrics:", metrics)
    trainer.save_model("gpu_finetuned_legacy/final")
    print("✅ Done. Model saved to gpu_finetuned_legacy/final")
