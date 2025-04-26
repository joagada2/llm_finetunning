#!/usr/bin/env python3
"""
gpu_finetune_legacy.py

Legacy-style fine-tuning of stabilityai/stablelm-base-alpha-7b on SST-2:
  • HF caches → ./hf_cache/
  • DeepSpeed ZeRO2 offload + fp16 + checkpointing
  • LoRA on NeoX modules
  • Legacy Trainer args (eval_steps, save_steps)
"""

import os, json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# ────────────────────────────────────────────────────────────────────────────────
# 0) Redirect ALL HF caches to a local folder
# ────────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
HF_CACHE = BASE / "hf_cache"
for sub in ("transformers", "datasets", "metrics", "hub"):
    (HF_CACHE / sub).mkdir(parents=True, exist_ok=True)

os.environ.update({
    "HF_HOME":            str(HF_CACHE / "hub"),
    "TRANSFORMERS_CACHE": str(HF_CACHE / "transformers"),
    "HF_DATASETS_CACHE":  str(HF_CACHE / "datasets"),
    "HF_METRICS_CACHE":   str(HF_CACHE / "metrics"),
    "HF_HUB_CACHE":       str(HF_CACHE / "hub"),
    "WANDB_DISABLED":     "true",           # disable wandb logging
})

# ────────────────────────────────────────────────────────────────────────────────
# 1) Paths & Seed
# ────────────────────────────────────────────────────────────────────────────────
set_seed(42)

DATA_ROOT  = BASE / "finetune_data"
MODEL_DIR  = DATA_ROOT / "model"
TRAIN_FILE = DATA_ROOT / "train.jsonl"
VALID_FILE = DATA_ROOT / "validation.jsonl"
MAX_LEN    = 128

# ────────────────────────────────────────────────────────────────────────────────
# 2) DeepSpeed config dump
# ────────────────────────────────────────────────────────────────────────────────
ds_conf = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"},
        "offload_param":     {"device": "cpu"},
    },
    "train_batch_size":             16,   # total effective batch
    "gradient_accumulation_steps":  2,
    "fp16":                         {"enabled": True},
}
with open("ds_config.json", "w") as f:
    json.dump(ds_conf, f, indent=2)

# ────────────────────────────────────────────────────────────────────────────────
# 3) Load & patch Model + Tokenizer
# ────────────────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=2,
    ignore_mismatched_sizes=True
)
model.config.pad_token_id = tokenizer.eos_token_id
model.resize_token_embeddings(len(tokenizer))

# LoRA on NeoX fused modules
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value", "dense"],
)
model = get_peft_model(model, lora_cfg)

# ────────────────────────────────────────────────────────────────────────────────
# 4) Dataset preparation
# ────────────────────────────────────────────────────────────────────────────────
raw = load_dataset(
    "json",
    data_files={"train": str(TRAIN_FILE), "validation": str(VALID_FILE)},
)

def preprocess(batch):
    toks = tokenizer(
        batch["sentence"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )
    toks["labels"] = batch["label"]
    return toks

tokenized = raw.map(
    preprocess,
    batched=True,
    num_proc=os.cpu_count(),
    remove_columns=raw["train"].column_names,
)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ────────────────────────────────────────────────────────────────────────────────
# 5) Metrics function
# ────────────────────────────────────────────────────────────────────────────────
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=-1)
    return {"accuracy": accuracy_score(pred.label_ids, preds)}

# ────────────────────────────────────────────────────────────────────────────────
# 6) Legacy TrainingArguments
# ────────────────────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="gpu_finetuned_legacy",
    logging_dir="logs_legacy",
    num_train_epochs=3,
    per_device_train_batch_size=4,      # per-GPU
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
    # note: no evaluation_strategy or save_strategy
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ────────────────────────────────────────────────────────────────────────────────
# 7) Train & evaluate
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    trainer.train()
    print(">> running final evaluation")
    metrics = trainer.evaluate(eval_dataset=tokenized["validation"])
    print("Validation metrics:", metrics)
    trainer.save_model("gpu_finetuned_legacy/final")
    print("✅ Done. Model saved to gpu_finetuned_legacy/final")
