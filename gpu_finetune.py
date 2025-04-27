 # ── 3) Load & patch tokenizer + model ──────────────────────────────────────
- tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
- tokenizer.pad_token = tokenizer.eos_token
-
- tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-base-alpha-7b", use_fast=True)
- model     = AutoModelForSequenceClassification.from_pretrained(
-     "stabilityai/stablelm-base-alpha-7b",
-     num_labels=2,
-     ignore_mismatched_sizes=True,
- )
+ tokenizer = AutoTokenizer.from_pretrained(
+     MODEL_DIR,
+     use_fast=True,
+     local_files_only=True
+ )
+ tokenizer.pad_token = tokenizer.eos_token
+
+ model = AutoModelForSequenceClassification.from_pretrained(
+     MODEL_DIR,
+     num_labels=2,
+     ignore_mismatched_sizes=True,
+     local_files_only=True
+ )

 # ensure caching disabled
 model.config.use_cache     = False
 model.config.pad_token_id  = tokenizer.eos_token_id
 model.resize_token_embeddings(len(tokenizer))

 # LoRA on NeoX modules
 lora_cfg = LoraConfig(
     r=16,
     lora_alpha=32,
     target_modules=["query_key_value", "dense"],
 )
 model = get_peft_model(model, lora_cfg)

 ...

-from transformers import Trainer as HfTrainer
+
+# import the base Trainer class
+from transformers import Trainer as HfTrainer

 # ── Custom Trainer to handle labels ────────────────────────────────────────
-class MyTrainer(HfTrainer):
-    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
-        labels = inputs.pop("labels")
-        outputs = model(**inputs, labels=labels)
-        loss = outputs.loss
-        return (loss, outputs) if return_outputs else loss
+class MyTrainer(HfTrainer):
+    # accept any extra kwargs (e.g. num_items_in_batch)
+    def compute_loss(self, model, inputs, *args, **kwargs):
+        labels = inputs.pop("labels")
+        outputs = model(**inputs, labels=labels)
+        loss = outputs.loss
+        if kwargs.get("return_outputs", False):
+            return loss, outputs
+        return loss
