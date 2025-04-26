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

    # Handle gated HF models auth
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN not set. Please 'huggingface-cli login' or export HF_TOKEN.", file=sys.stderr)
        sys.exit(1)

    # Model and dataset config
    model_name   = os.getenv("BASE_MODEL", "meta-llama/Llama-2-7b-chat-hf")
    dataset_name = "glue"
    subset_name  = "sst2"

    # 1) Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_auth_token=hf_token,
        )
    except Exception as e:
        print(f"Error loading tokenizer for {model_name}: {e}", file=sys.stderr)
        sys.exit(1)

    # 2) Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            use_auth_token=hf_token,
        )
    except Exception as e:
        print(f"Error loading model {model_name}: {e}", file=sys.stderr)
        sys.exit(1)

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

    # 5) Train/Val split
    split = tokenized.train_test_split(test_size=0.1)

    # 6) Training arguments
    training_args = TrainingArguments(
        output_dir="./llama2-7b-sst2-lora",
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
