#!/usr/bin/env python3
import os
import sys
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def tokenize_function(example, tokenizer):
    prompt = f"Review: {example['sentence']} Sentiment:"
    label_text = " Positive" if example['label'] == 1 else " Negative"
    result = tokenizer(
        prompt + label_text,
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    result["labels"] = result["input_ids"].copy()
    return result

def main():
    # Check for GPU
    print("CUDA available?", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("CUDA device name:", torch.cuda.get_device_name(0))
    else:
        print("No CUDA devices found. Training will run on CPU.")

    # Model and dataset config (no HF token required for ungated models)
    model_name   = os.getenv("BASE_MODEL", "EleutherAI/gpt-neo-1.3B")
    dataset_name = os.getenv("DATASET_NAME", "glue")
    subset_name  = os.getenv("DATASET_CONFIG", "sst2")

    # 1) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        truncation_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    # 2) Load model (no accelerator/triton issues)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
    )
    if torch.cuda.is_available():
        model = model.half().cuda()

    # 3) Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 4) Load & tokenize dataset
    dataset = load_dataset(dataset_name, subset_name)
    tokenized = dataset["train"].map(
        lambda ex: tokenize_function(ex, tokenizer),
        batched=False,
    )

    # 5) Split for evaluation
    split = tokenized.train_test_split(test_size=0.1)

    # 6) Training arguments
    training_args = TrainingArguments(
        output_dir=f"./{model_name.split('/')[-1]}-sst2-lora",
        do_eval=True,
        eval_steps=50,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=4,
    )

    # 7) Setup Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )
    trainer.train()

if __name__ == "__main__":
    main()
