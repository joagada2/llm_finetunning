#!/usr/bin/env python3
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
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
    model_name   = "Qwen/Qwen1.5-7B-Chat"
    dataset_name = "glue"
    subset_name  = "sst2"

    # 1) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # 2) Load model with NO bitsandbytes quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=False,
        load_in_8bit=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
    )

    # 3) Prepare model for k-bit (LoRA) training
    model = prepare_model_for_kbit_training(model)

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

    # 5) Split for eval
    split = tokenized.train_test_split(test_size=0.1)

    # 6) Training args
    training_args = TrainingArguments(
        output_dir="./qwen2.5_sst2_lora",
        evaluation_strategy="steps",
        eval_steps=50,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        save_total_limit=1,
        report_to="none",
    )

    # 7) Trainer & train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )
    trainer.train()

if __name__ == "__main__":
    main()
