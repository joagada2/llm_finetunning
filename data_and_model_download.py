#!/usr/bin/env python3
import os

from huggingface_hub import snapshot_download
from datasets import load_dataset

def download_model_and_dataset(
    model_name: str,
    dataset_name: str,
    dataset_config: str | None = None
):
    # 1) Download the tokenizer repo (files only)
    print(f"Downloading tokenizer for {model_name}...")
    tokenizer_cache = snapshot_download(
        repo_id=model_name,
        subfolder="tokenizer",
        use_auth_token=os.getenv("HF_TOKEN"),
    )

    # 2) Download the model repo (files only)
    print(f"Downloading model repository {model_name}...")
    model_cache = snapshot_download(
        repo_id=model_name,
        subfolder=None,
        use_auth_token=os.getenv("HF_TOKEN"),
    )

    # 3) Download the dataset
    cfg = f", config='{dataset_config}'" if dataset_config else ""
    print(f"Downloading dataset {dataset_name}{cfg}...")
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config)
    else:
        dataset = load_dataset(dataset_name)

    print("✅ Files downloaded (model & dataset). \n" \
          f"Tokenizer files in: {tokenizer_cache}\n" \
          f"Model files in: {model_cache}")
    return tokenizer_cache, model_cache, dataset

if __name__ == "__main__":
    # Set defaults or override via environment
    model_name   = os.getenv("BASE_MODEL", "distilgpt2")
    dataset_name = os.getenv("DATASET_NAME", "trl-lib/Capybara")
    dataset_cfg  = os.getenv("DATASET_CONFIG", None)

    download_model_and_dataset(model_name, dataset_name, dataset_cfg)
