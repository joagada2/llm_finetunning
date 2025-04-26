#!/usr/bin/env python3
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def get_latest_checkpoint(output_dir: str) -> str:
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {output_dir}")
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    latest = os.path.join(output_dir, checkpoints[-1])
    print(f"Latest checkpoint found: {latest}")
    return latest

from transformers import BitsAndBytesConfig

def load_model(
    base_model_name: str,
    lora_weights_path: str | None = None
):
    # … tokenizer as before …

    # ❶ Tell Transformers to load the base model in 4-bit
    bnb_conf = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,   # optional, defaults are fine
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_conf,    # ← 4-bit enabled
    )

    # ❷ Now apply your LoRA adapters and merge
    if lora_weights_path:
        model = PeftModel.from_pretrained(model, lora_weights_path, torch_dtype=torch.float16)
        model = model.merge_and_unload()

    return tokenizer, model

def predict_sentiment(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_new_tokens: int = 10
) -> str:
    device = model.device
    inputs = tokenizer(f"Review: {text} Sentiment:", return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    base_model_name = "Qwen/Qwen1.5-7B-Chat"
    output_dir      = "./qwen2.5_sst2_lora"

    example_texts = [
        "The movie was terrible and boring.",
        "I absolutely loved the acting and the storyline!",
    ]

    # --- Before fine-tuning ---
    print("=== Base Model Predictions ===")
    tokenizer, model = load_model(base_model_name)
    for txt in example_texts:
        print(f"Input:  {txt}")
        print(f"Output: {predict_sentiment(txt, tokenizer, model)}\n")

    # --- After fine-tuning ---
    print("=== Fine-Tuned Model Predictions ===")
    ckpt = get_latest_checkpoint(output_dir)
    tokenizer_ft, model_ft = load_model(base_model_name, lora_weights_path=ckpt)
    for txt in example_texts:
        print(f"Input:  {txt}")
        print(f"Output: {predict_sentiment(txt, tokenizer_ft, model_ft)}\n")

if __name__ == "__main__":
    main()
