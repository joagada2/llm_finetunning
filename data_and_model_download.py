#!/usr/bin/env python3
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def download_model_and_dataset(
    model_name: str,
    dataset_name: str,
    dataset_config: str | None = None
):
    # 1) Download the tokenizer
    print(f"Downloading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # 2) Download the model (fp16/fp32)
    print(f"Downloading model {model_name} (fp16/fp32)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # 3) Download the dataset
    cfg = f", config='{dataset_config}'" if dataset_config else ""
    print(f"Downloading dataset {dataset_name}{cfg}...")
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config)
    else:
        dataset = load_dataset(dataset_name)

    print("✅ Done!")
    return tokenizer, model, dataset


if __name__ == "__main__":
    # Set defaults or override via environment
    model_name   = os.getenv("BASE_MODEL", "distilgpt2")
    dataset_name = os.getenv("DATASET_NAME", "trl-lib/Capybara")
    dataset_cfg  = os.getenv("DATASET_CONFIG", None)

    download_model_and_dataset(model_name, dataset_name, dataset_cfg)
