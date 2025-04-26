#!/usr/bin/env python3
"""
gpu_finetune_complete.py

All-in-one script for finetuning stabilityai/stablelm-base-alpha-7b on SST-2:
- Redirects ALL HF caches to ./hf_cache/ before any imports
- Uses DeepSpeed Stage2 offload, fp16, gradient checkpointing
- Applies LoRA on GPT-NeoX modules
- Legacy Trainer with eval_steps/save_steps
- Avoids home-dir quota errors
"""

import os
from pathlib import Path

# 0) Redirect ALL HF caches immediately before any HF imports
BASE = Path(__file__).parent
HF_CACHE = BASE / "hf_cache"
for sub in ("hub", "transformers", "datasets", "metrics"):
    (HF_CACHE / sub).mkdir(parents=True, exist_ok=True)

os.environ.update({
    "HF_HOME":            str(HF_CACHE / "hub"),
    "TRANSFORMERS_CACHE": str(HF_CACHE / "transformers"),
    "HF_DATASETS_CACHE":  str(HF_CACHE / "datasets"),
    "HF_METRICS_CACHE":   str(HF_CACHE / "metrics"),
    "WANDB_DISABLED":     "true",
})

import json
import numpy as np
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

# 1) Paths & seed
set_seed(42)
DATA_ROOT  = BASE / "finetune_data"
MODEL_DIR  = DATA_ROOT / "model"
TRAIN_FILE = DATA_ROOT / "train.jsonl"
VALID_FILE = DATA_ROOT / "validation.jsonl"
MAX_LEN    = 128

# 2) DeepSpeed config
ds_conf = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"},
        "offload_param":     {"device": "cpu"},
    },
    "train_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "fp16": {"enabled": True},
}
with open("ds_config.json", "w") as f:
    json.dump(ds_conf, f, indent=2)

# 3) Load & patch tokenizer/model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, num_labels=2, ignore_mismatched_sizes=True
)
model.config.pad_token_id = tokenizer.eos_token_id
model.resize_token_embeddings(len(tokenizer))

# Apply LoRA
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value", "dense"],
)
model = get_peft_model(model, lora_cfg)

# 4) Dataset prep
raw = load_dataset(
    "json",
    data_files={"train": str(TRAIN_FILE), "validation": str(VALID_FILE)},
    cache_dir=str(HF_CACHE / "datasets")
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

# 5) Metrics
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=-1)
    return {"accuracy": accuracy_score(pred.label_ids, preds)}

# 6) TrainingArguments (legacy)
training_args = TrainingArguments(
    output_dir="gpu_finetuned_complete",
    logging_dir="logs_complete",
    num_train_epochs=3,
    per_device_train_batch_size=4,
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
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 7) Train & evaluate
if __name__ == "__main__":
    trainer.train()
    print(">> Running final evaluation")
    metrics = trainer.evaluate(eval_dataset=tokenized["validation"])
    print("Validation metrics:", metrics)
    trainer.save_model("gpu_finetuned_complete/final")
    print("✅ Done. Model saved to gpu_finetuned_complete/final")
