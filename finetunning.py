#!/usr/bin/env python3
import os
import sys
import types
import importlib.machinery

# Stub out Triton to avoid driver errors on CPU-only nodes
triton_mod = types.ModuleType('triton')
triton_mod.Config = lambda *args, **kwargs: None
triton_mod.runtime = types.SimpleNamespace(driver=types.SimpleNamespace(active=[]))
# Provide a valid __spec__ so import_utils.find_spec sees the module
triton_mod.__spec__ = importlib.machinery.ModuleSpec('triton', None)
sys.modules['triton'] = triton_mod

# Stub out torchao to prevent import errors on CPU-only nodes
torchao_mod = types.ModuleType('torchao')
torchao_mod.__spec__ = importlib.machinery.ModuleSpec('torchao', None)
kernel_mod = types.ModuleType('torchao.kernel')
kernel_mod.intmm_triton = lambda *args, **kwargs: None
kernel_mod.__spec__ = importlib.machinery.ModuleSpec('torchao.kernel', None)
sys.modules['torchao'] = torchao_mod
sys.modules['torchao.kernel'] = kernel_mod

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)

# Hyperparameters
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 4))
EPOCHS = int(os.getenv("EPOCHS", 1))
LR = float(os.getenv("LEARNING_RATE", 2e-4))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 256))
GRAD_ACC_STEPS = int(os.getenv("GRAD_ACC_STEPS", 1))
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", 10))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./finetuned-model")

# Tokenization

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

# Training loop
def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Load model & tokenizer
    model_name = os.getenv("BASE_MODEL", "EleutherAI/gpt-neo-1.3B")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.train()

    # Load and tokenize dataset
    dataset = load_dataset('glue', 'sst2', split='train')
    tokenized = dataset.map(lambda ex: tokenize_function(ex, tokenizer), batched=False)

    # Data loader
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    dataloader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Training
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

    # Save model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned model to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
