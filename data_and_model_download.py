#!/usr/bin/env python3
import sys
import types
import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Stub out deepspeed and triton to prevent import errors on CPU-only nodes
sys.modules['deepspeed'] = types.ModuleType('deepspeed')
triton_mod = types.ModuleType('triton')
triton_mod.Config = lambda *args, **kwargs: None
triton_mod.jit = lambda *args, **kwargs: (lambda fn: fn)
triton_mod.language = types.SimpleNamespace(jit=lambda *a, **k: None)
sys.modules['triton'] = triton_mod
sys.modules['triton.language'] = types.ModuleType('triton.language')


def download_model_and_data(
    model_name: str,
    dataset_name: str,
    dataset_config: str | None = None,
    subset_pct: float = 0.1
):
    """
    Download a HuggingFace-compatible Llama model, prepare tokenizer/model, and load a dataset subset.

    Args:
        model_name: HF repo ID for the model (must include valid config.json).
        dataset_name: HF dataset name.
        dataset_config: Optional config name (e.g. 'sst2').
        subset_pct: Fraction of the training split to download.
    """
    # 1) Download entire model repo locally
    print(f"Downloading model files for {model_name}...")
    repo_dir = snapshot_download(model_name)
    print(f"Model files saved to: {repo_dir}")

    # 2) Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(repo_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) Load model into GPU in fp16
    print("Loading model (fp16, device_map=auto)...")
    model = AutoModelForCausalLM.from_pretrained(
        repo_dir,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # 4) Load dataset subset
    percent = int(subset_pct * 100)
    split = f"train[:{percent}%]"
    ds_args = {"split": split}
    if dataset_config:
        ds_args.update({"name": dataset_config})
    print(f"Loading dataset {dataset_name}/{dataset_config} split={split}...")
    dataset = load_dataset(dataset_name, **ds_args)

    print("Download complete.")
    return tokenizer, model, dataset


if __name__ == "__main__":
    # Use a HF repo with valid config.json (no '-GGUF' suffix)
    MODEL = os.getenv("BASE_MODEL", "VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct")
    DATASET = os.getenv("DATASET_NAME", "glue")
    CONFIG = os.getenv("DATASET_CONFIG", "sst2")

    tok, mod, ds = download_model_and_data(
        MODEL,
        DATASET,
        CONFIG,
        subset_pct=0.1,
    )

    # Save locally for finetuning
    print("Saving tokenizer and model to ./downloaded-model/")
    mod.save_pretrained("./downloaded-model")
    tok.save_pretrained("./downloaded-model")
    print("Saving dataset to ./downloaded-data/")
    ds.save_to_disk("./downloaded-data")

    print("✅ Download complete.")
