#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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

    # 2) Download the model (no bitsandbytes quantization)
    print(f"Downloading model {model_name} (fp16/fp32)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=None,      # ← disable all bnb quant
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
    # You can change these to whatever you need
    model_name   = "Qwen/Qwen2.5-0.5B"
    dataset_name = "trl-lib/Capybara"
    dataset_cfg  = None     # e.g. "multiturn" if that HF dataset has a config

    download_model_and_dataset(model_name, dataset_name, dataset_cfg)
