import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def get_latest_checkpoint(output_dir):
    checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint-")]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {output_dir}")
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    latest_checkpoint = os.path.join(output_dir, checkpoints[-1])
    print(f"Latest checkpoint found: {latest_checkpoint}")
    return latest_checkpoint

def load_model(base_model_name, lora_weights_path=None):
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True
    )

    if lora_weights_path:
        print(f"Applying LoRA weights from: {lora_weights_path}")
        model = PeftModel.from_pretrained(model, lora_weights_path)
        model = model.merge_and_unload()

    return tokenizer, model

def predict_sentiment(text, tokenizer, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    inputs = tokenizer(f"Review: {text} Sentiment:", return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    base_model_name = "Qwen/Qwen1.5-7B-Chat"
    output_dir = "./qwen2.5_sst2_lora"

    example_texts = [
        "The movie was terrible and boring.",
        "I absolutely loved the acting and the storyline!",
    ]

    # Before Finetuning
    print("=== Base Model Predictions ===")
    tokenizer, model = load_model(base_model_name)
    for text in example_texts:
        print(f"Input: {text}")
        print(f"Output: {predict_sentiment(text, tokenizer, model)}\n")

    # After Finetuning
    print("\n=== Finetuned Model Predictions ===")
    latest_checkpoint = get_latest_checkpoint(output_dir)
    tokenizer, model = load_model(base_model_name, lora_weights_path=latest_checkpoint)
    for text in example_texts:
        print(f"Input: {text}")
        print(f"Output: {predict_sentiment(text, tokenizer, model)}\n")

if __name__ == "__main__":
    main()
