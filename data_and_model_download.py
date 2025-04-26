#!/usr/bin/env python3
import os
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset

def download_model_and_data(
    model_name: str,
    dataset_name: str,
    dataset_config: str,
    subset_pct: float = 0.1
):
    """
    1) Pull down a HF‐compatible LLaMA‐Instruct repo
    2) Load its tokenizer & model (fp16 → GPU via device_map="auto")
    3) Load a small subset of the dataset
    """

    # 1) Tokenizer
    print(f"Downloading tokenizer for {model_name}…")
    tokenizer = LlamaTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    # ensure pad_token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) Model
    print(f"Downloading model {model_name} (fp16 → GPU)…")
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # 3) Dataset
    pct = int(subset_pct * 100)
    split = f"train[:{pct}%]"
    print(f"Loading dataset {dataset_name}/{dataset_config} split={split}…")
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    return tokenizer, model, dataset


if __name__ == "__main__":
    MODEL   = os.getenv("BASE_MODEL",   "VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct")
    DATASET = os.getenv("DATASET_NAME", "glue")
    CONFIG  = os.getenv("DATASET_CONFIG", "sst2")

    tok, mod, ds = download_model_and_data(MODEL, DATASET, CONFIG, subset_pct=0.1)

    # save to disk for finetuning
    print("Saving tokenizer and model to ./downloaded-model/")
    mod.save_pretrained("./downloaded-model")
    tok.save_pretrained("./downloaded-model")

    print("Saving dataset to ./downloaded-data/")
    ds.save_to_disk("./downloaded-data")

    print("✅ Download complete.")
