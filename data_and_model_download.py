#!/usr/bin/env python3
import os

from huggingface_hub import snapshot_download
from datasets import load_dataset

def download_model_and_dataset(
    model_name: str,
    dataset_name: str,
    dataset_config: str | None = None
):
    # Download the model repository (includes tokenizer and model files)
    print(f"Downloading all files for {model_name}...")
    repo_cache = snapshot_download(
        repo_id=model_name,
        use_auth_token=os.getenv("HF_TOKEN") or None,
    )

    # Inform user where files are
    print("✅ Model & tokenizer files downloaded.")
    print(f"Local cache directory: {repo_cache}\n")

    # Download the dataset
    cfg = f", config='{dataset_config}'" if dataset_config else ""
    print(f"Downloading dataset {dataset_name}{cfg}...")
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config)
    else:
        dataset = load_dataset(dataset_name)

    print("✅ Dataset downloaded.")
    return repo_cache, dataset

if __name__ == "__main__":
    model_name   = os.getenv("BASE_MODEL", "distilgpt2")
    dataset_name = os.getenv("DATASET_NAME", "trl-lib/Capybara")
    dataset_cfg  = os.getenv("DATASET_CONFIG", None)

    download_model_and_dataset(model_name, dataset_name, dataset_cfg)
