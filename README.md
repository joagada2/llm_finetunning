# Fine-tuning Qwen2.5-7B on SST2 (Sentiment Classification)

This project fine-tunes the HuggingFace-hosted `Qwen/Qwen1.5-7B-Chat` model using the labeled SST2 dataset (`glue/sst2`) for binary sentiment classification.

---

## 📦 Requirements

```bash
pip install torch transformers datasets peft accelerate bitsandbytes
