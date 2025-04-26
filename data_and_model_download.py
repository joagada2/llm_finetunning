#!/usr/bin/env python3
import sys
import types
import importlib.machinery

# Stub out deepspeed to prevent import errors on CPU-only nodes
import sys
import types
import importlib.machinery

deep_pkg = types.ModuleType('deepspeed')
deep_pkg.__spec__ = importlib.machinery.ModuleSpec('deepspeed', None)
class DeepSpeedEngine: pass
deep_pkg.DeepSpeedEngine = DeepSpeedEngine
deep_ops = types.ModuleType('deepspeed.ops')
deep_ops.__spec__ = importlib.machinery.ModuleSpec('deepspeed.ops', None)
deep_pkg.ops = deep_ops
sys.modules['deepspeed'] = deep_pkg
sys.modules['deepspeed.ops'] = deep_ops

# Stub out Triton to prevent import errors on CPU-only nodes
triton_mod = types.ModuleType('triton')
triton_mod.Config = lambda *args, **kwargs: None
triton_mod.jit = lambda *args, **kwargs: (lambda fn: fn)
triton_mod.runtime = types.SimpleNamespace(driver=types.SimpleNamespace(active=[]))
triton_mod.__spec__ = importlib.machinery.ModuleSpec('triton', None)
sys.modules['triton'] = triton_mod

# Stub Triton.language submodule for torchao dependencies
import importlib
tl_mod = types.ModuleType('triton.language')
tl_mod.__spec__ = importlib.machinery.ModuleSpec('triton.language', None)
tl_mod.constexpr = lambda x: x
tl_mod.int32 = None
tl_mod.int64 = None
tl_mod.float16 = None
tl_mod.float32 = None
sys.modules['triton.language'] = tl_mod

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
    Download a model and a subset of a Hugging Face dataset.

    Args:
        model_name: HF repo ID for the model.
        dataset_name: HF dataset name.
        dataset_config: Optional config name (e.g., 'sst2' for GLUE).
        subset_pct: Fraction of the training split to download.
    """
    # 1) Download tokenizer
    print(f"Downloading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if hasattr(tokenizer, 'eos_token') else tokenizer.all_special_tokens[-1]

    # 2) Download model in FP16 on GPU
    print(f"Downloading model {model_name} (FP16, device_map=auto)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # 3) Load dataset subset
    split_str = f"train[:{int(subset_pct*100)}%]"
    suffix = f"/{dataset_config}" if dataset_config else ""
    print(f"Loading dataset {dataset_name}{suffix} split={split_str}...")
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split_str)
    else:
        dataset = load_dataset(dataset_name, split=split_str)

    print("Download complete.")
    return tokenizer, model, dataset

if __name__ == "__main__":
    MODEL = os.getenv("BASE_MODEL", "QuantFactory/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF")
    DATASET = os.getenv("DATASET_NAME", "glue")
    CONFIG = os.getenv("DATASET_CONFIG", "sst2")
    tok, mod, ds = download_model_and_data(MODEL, DATASET, CONFIG, subset_pct=0.1)
    mod.save_pretrained("./downloaded-model")
    tok.save_pretrained("./downloaded-model")
    ds.save_to_disk("./downloaded-data")
