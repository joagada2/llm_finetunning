#!/usr/bin/env python3
"""
finetune_stablelm_optimized.py

Optimized fine-tuning of stabilityai/stablelm-base-alpha-7b on SST-2:
- GPU + fp16 + gradient_checkpointing
- DeepSpeed ZeRO2 offload
- LoRA for PEFT
- Multi-proc tokenization
- Disabled wandb + local hf_cache
"""

import os, json
from pathlib import Path

# ── 0) ENVIRONMENT ───────────────────────────────────────────────────────────
BASE_PATH = Path(__file__).parent
# Redirect HF caches to local folder
HF_CACHE = BASE_PATH / "hf_cache"
for sub in ("transformers","datasets","metrics","hub"):
    (HF_CACHE/sub).mkdir(parents=True, exist_ok=True)
os.environ.update({
    "HF_HOME":             str(HF_CACHE/"hub"),
    "TRANSFORMERS_CACHE":  str(HF_CACHE/"transformers"),
    "HF_DATASETS_CACHE":   str(HF_CACHE/"datasets"),
    "HF_METRICS_CACHE":    str(HF_CACHE/"metrics"),
    "HF_HUB_CACHE":        str(HF_CACHE/"hub"),
    "WANDB_DISABLED":      "true",
})

# ── 1) Imports (after env setup) ─────────────────────────────────────────────
import numpy as np
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model

# ── 2) CONFIG ─────────────────────────────────────────────────────────────────
set_seed(42)
BASE_DIR       = BASE_PATH/"finetune_data"
MODEL_DIR      = BASE_DIR/"model"
TRAIN_FILE     = BASE_DIR/"train.jsonl"
VALID_FILE     = BASE_DIR/"validation.jsonl"
MAX_LENGTH     = 128
OUTPUT_DIR     = "stablelm_zero2_lora"
LOG_DIR        = "logs_optimized"

# DeepSpeed ZeRO2 config
DS_CONFIG = {
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "cpu" },
    "offload_param":     { "device": "cpu" }
  },
  "train_batch_size":           16,
  "gradient_accumulation_steps": 2,
  "fp16":                       { "enabled": True }
}
# dump DS config
with open("ds_config.json","w") as f:
    json.dump(DS_CONFIG, f, indent=2)

# ── 3) Load & patch tokenizer/model ─────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# base model: loads config
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, num_labels=2, ignore_mismatched_sizes=True
)
model.config.pad_token_id = tokenizer.eos_token_id
model.resize_token_embeddings(len(tokenizer))

# apply LoRA (PEFT)
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value", "dense"]
)

model = get_peft_model(model, lora_cfg)

# ── 4) Prepare & tokenize dataset ────────────────────────────────────────────
raw = load_dataset(
    "json", data_files={"train": str(TRAIN_FILE), "validation": str(VALID_FILE)}
)

def preprocess(batch):
    toks = tokenizer(
        batch["sentence"],
        truncation=True, padding="max_length",
        max_length=MAX_LENGTH,
    )
    toks["labels"] = batch["label"]
    return toks

tokenized = raw.map(
    preprocess,
    batched=True,
    num_proc=os.cpu_count(),
    remove_columns=raw["train"].column_names
)
tokenized.set_format("torch", columns=["input_ids","attention_mask","labels"])

# ── 5) Metrics fn ────────────────────────────────────────────────────────────
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=-1)
    return {"accuracy": accuracy_score(pred.label_ids, preds)}

# ── 6) TrainingArguments ─────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=LOG_DIR,
    per_device_train_batch_size=4,    # per-GPU
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,    # effective batch = 8
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    gradient_checkpointing=True,
    deepspeed="ds_config.json",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
    save_total_limit=2,
    seed=42,
)

# ── 7) Trainer & train ───────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

if __name__=="__main__":
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/best")
    print("✅ Optimized fine-tuning complete. Model at", f"{OUTPUT_DIR}/best")
