#!/usr/bin/env python3
import os
import sys
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

# Hyperparameters
BATCH_SIZE = 2
EPOCHS = 1
LR = 2e-4
MAX_LENGTH = 256
GRAD_ACC_STEPS = 4
LOGGING_STEPS = 10
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./gpt-neo-1.3B-sst2-lora")

# Tokenization function for supervised fine-tuning
def tokenize_function(example, tokenizer):
    prompt = f"Review: {example['sentence']} Sentiment:"
    label_text = " Positive" if example['label'] == 1 else " Negative"
    encoded = tokenizer(
        prompt + label_text,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    encoded['labels'] = encoded['input_ids'].clone()
    return encoded

# Main training loop
def main():
    # 1) Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 2) Load model & tokenizer
    model_name = os.getenv("BASE_MODEL", "EleutherAI/gpt-neo-1.3B")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    # 3) Prepare LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    # 4) Load and tokenize dataset
    dataset = load_dataset('glue', 'sst2', split='train')
    tokenized = dataset.map(lambda ex: tokenize_function(ex, tokenizer), batched=False)

    # 5) DataLoader
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    dataloader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)

    # 6) Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 7) Training
    global_step = 0
    for epoch in range(EPOCHS):
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / GRAD_ACC_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACC_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % LOGGING_STEPS == 0:
                    print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item() * GRAD_ACC_STEPS:.4f}")

    # 8) Save LoRA adapters
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved LoRA-adapted model to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
