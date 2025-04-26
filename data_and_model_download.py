#!/usr/bin/env python3
import sys
import types
import importlib.machinery
# Stub out Triton to prevent import errors on CPU-only nodes
triton_mod = types.ModuleType('triton')
triton_mod.Config = lambda *args, **kwargs: None
triton_mod.runtime = types.SimpleNamespace(driver=types.SimpleNamespace(active=[]))
triton_mod.__spec__ = importlib.machinery.ModuleSpec('triton', None)
sys.modules['triton'] = triton_mod

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def download_model_and_data(
    model_name: str,
    dataset_name: str,
    dataset_config: str | None = None,
    subset_pct: float = 0.1
):
    """
    Download pretrained model and a subset of a Hugging Face dataset.

    Args:
        model_name: HF repo ID for the model.
        dataset_name: HF dataset name.
        dataset_config: Optional config name (e.g. "sst2" for GLUE).
        subset_pct: Fraction of the training split to download.
    """
    # 1) Download tokenizer
    print(f"Downloading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # 2) Download model (FP16 on GPU)
    print(f"Downloading model {model_name} (FP16, device_map=auto)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # 3) Load a subset of the dataset
    split_str = f"train[:{int(subset_pct*100)}%]"
    print(f"Loading dataset {dataset_name}{f'/{dataset_config}' if dataset_config else ''} split={split_str}...")
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split_str)
    else:
        dataset = load_dataset(dataset_name, split=split_str)

    print("Download complete.")
    return tokenizer, model, dataset

if __name__ == "__main__":
    # Example usage
    MODEL = os.getenv("BASE_MODEL", "QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF")
    DATASET = os.getenv("DATASET_NAME", "glue")
    CONFIG = os.getenv("DATASET_CONFIG", "sst2")
    TOK, MOD, DS = download_model_and_data(MODEL, DATASET, CONFIG, subset_pct=0.1)
    # Save locally
    MOD.save_pretrained("./downloaded-model")
    TOK.save_pretrained("./downloaded-model")
    DS.save_to_disk("./downloaded-data")
