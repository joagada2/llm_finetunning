from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    trust_remote_code=True,
    device_map="auto",
    quantization_config=None,   # ← disable all bnb quantization
)

    # Download model
    print("Downloading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        load_in_4bit=True  # Optional: saves memory
    )

    # Download dataset (labeled) from HuggingFace
    print("Downloading labeled dataset...")
    dataset = load_dataset(dataset_name, subset_name)

    print("Done!")

if __name__ == "__main__":
    download_model_and_dataset()
