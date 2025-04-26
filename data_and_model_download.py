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

    # 2) Download the model (fp16/fp32)
    print(f"Downloading model {model_name} (fp16/fp32)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=None,      # ← no bnb quant
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
    # Change this to the LLaMA variant you prefer:
    #  - LLaMA-2 7B chat:  "meta-llama/Llama-2-7b-chat-hf"
    #  - LLaMA-2 8B chat:  "meta-llama/Llama-2-8b-chat-hf"
    # model_name   = "meta-llama/Llama-2-7b-chat-hf"
    model_name   = "EleutherAI/gpt-neo-1.3B"
    dataset_name = "trl-lib/Capybara"
    dataset_cfg  = None     # e.g. "multiturn" if that HF dataset has a config

    download_model_and_dataset(model_name, dataset_name, dataset_cfg)
