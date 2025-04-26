#!/usr/bin/env python3
import sys
import types
import importlib.machinery
import os
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Stub out deepspeed to prevent import errors on CPU-only nodes
sys.modules['deepspeed'] = types.ModuleType('deepspeed')
# Stub out Triton to prevent import errors on CPU-only nodes
triton_mod = types.ModuleType('triton')
triton_mod.Config = lambda *args, **kwargs: None
triton_mod.jit = lambda *args, **kwargs: (lambda fn: fn)
triton_mod.runtime = types.SimpleNamespace(driver=types.SimpleNamespace(active=[]))
sys.modules['triton'] = triton_mod
sys.modules['triton.language'] = types.ModuleType('triton.language')  # no-op


def download_model_and_data(
    model_name: str,
    dataset_name: str,
    dataset_config: str | None = None,
    subset_pct: float = 0.1
):
    """
    Download a model and a subset of a Hugging Face dataset.

    Args:
        model_name: HF repo ID for the model.
        dataset_name: HF dataset name.
        dataset_config: Optional config name (e.g., 'sst2' for GLUE).
        subset_pct: Fraction of the training split to download.
    """
    # 1) Load config (enables trust_remote_code hooks)
    print(f"Loading config for {model_name}...")
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # 2) Download tokenizer
    print(f"Downloading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (tokenizer.eos_token or tokenizer.all_special_tokens[-1])

    # 3) Download model in FP16 on GPU
    print(f"Downloading model {model_name} (FP16, device_map=auto)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # 4) Load dataset subset
    split_str = f"train[:{int(subset_pct * 100)}%]"
    print(f"Loading dataset {dataset_name}{('/' + dataset_config) if dataset_config else ''} split={split_str}...")
    if dataset_config:
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split_str,
        )
    else:
        dataset = load_dataset(
            dataset_name,
            split=split_str,
        )

    print("Download complete.")
    return tokenizer, model, dataset


if __name__ == "__main__":
    MODEL = os.getenv(
        "BASE_MODEL",
        "QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF",
    )
    DATASET = os.getenv("DATASET_NAME", "glue")
    CONFIG = os.getenv("DATASET_CONFIG", "sst2")

    tok, mod, ds = download_model_and_data(
        MODEL,
        DATASET,
        CONFIG,
        subset_pct=0.1,
    )

    mod.save_pretrained("./downloaded-model")
    tok.save_pretrained("./downloaded-model")
    ds.save_to_disk("./downloaded-data")
