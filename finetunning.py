#!/usr/bin/env python3
import os
import sys
import types
import importlib.machinery

# Stub out deepspeed to prevent import errors on CPU-only nodes
deep_pkg = types.ModuleType('deepspeed')
deep_pkg.__spec__ = importlib.machinery.ModuleSpec('deepspeed', None)
deep_ops = types.ModuleType('deepspeed.ops')
deep_ops.__spec__ = importlib.machinery.ModuleSpec('deepspeed.ops', None)
deep_pkg.ops = deep_ops
sys.modules['deepspeed'] = deep_pkg
sys.modules['deepspeed.ops'] = deep_ops

# Stub out Triton to avoid driver errors on CPU-only nodes
triton_mod = types.ModuleType('triton')
triton_mod.Config = lambda *args, **kwargs: None
triton_mod.runtime = types.SimpleNamespace(driver=types.SimpleNamespace(active=[]))
triton_mod.__spec__ = importlib.machinery.ModuleSpec('triton', None)
sys.modules['triton'] = triton_mod

# Stub out torchao and submodules to prevent import errors
torchao_mod = types.ModuleType('torchao')
torchao_mod.__spec__ = importlib.machinery.ModuleSpec('torchao', None)
kernel_mod = types.ModuleType('torchao.kernel')
kernel_mod.intmm_triton = lambda *args, **kwargs: None
kernel_mod.__spec__ = importlib.machinery.ModuleSpec('torchao.kernel', None)
sys.modules['torchao'] = torchao_mod
sys.modules['torchao.kernel'] = kernel_mod

# Stub out torchao.float8 and float8_linear
float8_pkg = types.ModuleType('torchao.float8')
float8_pkg.__spec__ = importlib.machinery.ModuleSpec('torchao.float8', None)
float8_linear_mod = types.ModuleType('torchao.float8.float8_linear')
class Float8LinearConfig: pass
float8_linear_mod.Float8LinearConfig = Float8LinearConfig
float8_linear_mod.__spec__ = importlib.machinery.ModuleSpec('torchao.float8.float8_linear', None)
sys.modules['torchao.float8'] = float8_pkg
sys.modules['torchao.float8.float8_linear'] = float8_linear_mod

# Stub out torchao.quantization
quant_mod = types.ModuleType('torchao.quantization')
quant_mod.__spec__ = importlib.machinery.ModuleSpec('torchao.quantization', None)
class Int4WeightOnlyConfig: pass
quant_mod.Int4WeightOnlyConfig = Int4WeightOnlyConfig
sys.modules['torchao.quantization'] = quant_mod

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Hyperparameters from environment or defaults
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 4))
EPOCHS = int(os.getenv("EPOCHS", 1))
LR = float(os.getenv("LEARNING_RATE", 2e-4))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 256))
GRAD_ACC_STEPS = int(os.getenv("GRAD_ACC_STEPS", 1))
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", 10))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./finetuned-model")

# Tokenize a single example for supervised fine-tuning
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
    # 1) Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 2) Select model based on available resources
    if not torch.cuda.is_available():
        print("No GPU detected – switching to a lightweight CPU-friendly model: distilgpt2")
        model_name = "distilgpt2"
    else:
        model_name = os.getenv("BASE_MODEL", "EleutherAI/gpt-neo-1.3B")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    # 3) Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # Disable use_cache to prevent 4D causal mask issues during training
    model.config.use_cache = False
    model.to(device)
    model.train()
    # Disable use_cache to prevent 4D causal mask issues during training
    model.config.use_cache = False
    model.to(device)
    model.train()

    # 3) Load and tokenize dataset
    dataset = load_dataset('glue', 'sst2', split='train')
    tokenized = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer),
        batched=False,
        remove_columns=dataset.column_names
    )

    # 4) DataLoader with padding collate
    def collate_fn(batch):
        return tokenizer.pad(
            batch,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
    dataloader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 5) Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 6) Training loop with gradient accumulation
    global_step = 0
    for epoch in range(EPOCHS):
        for step, batch in enumerate(dataloader):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, use_cache=False)  # disable caching to avoid 4D mask shape issues
            loss = outputs.loss / GRAD_ACC_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACC_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % LOGGING_STEPS == 0:
                    print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item() * GRAD_ACC_STEPS:.4f}")

    # 7) Save fine-tuned model and tokenizer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved fine-tuned model to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
