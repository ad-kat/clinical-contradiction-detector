"""
finetune_flan_t5.py
Fine-tunes google/flan-t5-small on MIMIC-IV derived contradiction_dataset.jsonl
using HuggingFace PEFT + LoRA. Runs on WSL2 CPU (overnight) or GPU if available.

Usage:
    python scripts/finetune_flan_t5.py
    python scripts/finetune_flan_t5.py --data data/contradiction_dataset.jsonl
    python scripts/finetune_flan_t5.py --model google/flan-t5-base  # if you have GPU

Requirements:
    pip install transformers peft datasets scikit-learn accelerate torch
"""

import json
import time
import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType
import torch

# ── Args ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data",  default="data/contradiction_dataset.jsonl")
parser.add_argument("--model", default="google/flan-t5-small")  # small = CPU-viable
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch",  type=int, default=8)
parser.add_argument("--output", default="models/flan-t5-clinical")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"Model:  {args.model}")
print(f"Data:   {args.data}")

# ── Load JSONL ──────────────────────────────────────────────────────────────
print("\nLoading dataset...")
data = []
with open(args.data) as f:
    for line in f:
        ex = json.loads(line.strip())
        # Support both formats:
        # - generate_training_data.py output (has top-level "input"/"output")
        # - raw label dict (has "text" + "label" keys)
        if "input" in ex and "output" in ex:
            data.append({"input": ex["input"], "output": ex["output"]})
        elif "text" in ex and "label" in ex:
            label = ex["label"]
            data.append({
                "input":  f"clinical contradiction detection: {ex['text']}",
                "output": f"contradiction:{label['contradiction']} type:{label['type']}"
            })
        else:
            continue  # skip malformed rows

print(f"Loaded {len(data)} examples")

# ── Train / val split (90/10) ───────────────────────────────────────────────
positives = [ex for ex in data if "contradiction:True" in ex["output"]]
data = data + positives
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
print(f"Train: {len(train_data)}  Val: {len(val_data)}")

# ── Tokenizer ───────────────────────────────────────────────────────────────
print(f"\nLoading tokenizer: {args.model}")
tokenizer = T5Tokenizer.from_pretrained(args.model)

MAX_INPUT  = 512   # note pairs can be long
MAX_OUTPUT = 32    # "contradiction:True type:allergy_medication"

def tokenize_batch(batch):
    model_inputs = tokenizer(
        batch["input"],
        max_length=MAX_INPUT,
        truncation=True,
        padding="max_length",
    )
    # T5 labels: tokenize targets separately
    labels = tokenizer(
        text_target=batch["output"],
        max_length=MAX_OUTPUT,
        truncation=True,
        padding="max_length",
        )
    # Replace padding token id with -100 so loss ignores padding
    label_ids = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = label_ids
    return model_inputs

train_ds = Dataset.from_list(train_data).map(tokenize_batch, batched=True, batch_size=64)
val_ds   = Dataset.from_list(val_data).map(tokenize_batch,   batched=True, batch_size=64)

# ── Model + LoRA ────────────────────────────────────────────────────────────
print(f"\nLoading model: {args.model}")
model = T5ForConditionalGeneration.from_pretrained(args.model)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    # flan-t5 attention projection names
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: ~0.6% trainable — e.g. "trainable params: 589,824 || all params: 77M"

# ── Training args ────────────────────────────────────────────────────────────
# CPU-safe settings: no fp16, small batch, gradient accumulation to simulate larger batch
use_fp16 = DEVICE == "cuda"
'''
training_args = TrainingArguments(
    output_dir=args.output,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=args.batch,
    gradient_accumulation_steps=4,          # effective batch = 8 * 4 = 32
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=False,
    bf16=False,
    logging_steps=25,
    warmup_ratio=0.1,
    weight_decay=0.01,
    report_to="none",                       # no wandb
    dataloader_num_workers=0,               # WSL2 safe
)
'''

training_args = TrainingArguments(
    output_dir=args.output,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=args.batch,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",        # was evaluation_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=use_fp16,
    logging_steps=25,
    warmup_ratio=0.1,
    weight_decay=0.01,
    report_to="none",
    dataloader_num_workers=0,
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
)

# ── Train ────────────────────────────────────────────────────────────────────
print("\nStarting training...")
print("flan-t5-small on CPU: ~2-4 hrs for 2000 examples, 3 epochs")
print("flan-t5-small on GPU: ~10-15 min\n")

train_start = time.time()
trainer.train()
train_elapsed = time.time() - train_start
print(f"\nTraining done in {train_elapsed/60:.1f} min")

# Save final model
trainer.save_model(args.output)
tokenizer.save_pretrained(args.output)
print(f"Model saved to {args.output}/")

# ── Evaluation: F1 + latency ─────────────────────────────────────────────────
print("\nEvaluating on validation set...")

def parse_contradiction(text: str) -> bool:
    """Extract boolean from 'contradiction:True type:...' """
    try:
        part = text.lower().split("contradiction:")[1].split()[0]
        return part.startswith("true")
    except (IndexError, AttributeError):
        return False

def parse_type(text: str) -> str:
    """Extract type label from output string."""
    try:
        return text.lower().split("type:")[1].strip().split()[0]
    except (IndexError, AttributeError):
        return "none"

model.eval()
preds_bool, labels_bool = [], []
preds_type, labels_type = [], []

latencies = []
for ex in val_data:
    inp = tokenizer(
        ex["input"],
        return_tensors="pt",
        max_length=MAX_INPUT,
        truncation=True,
    ).to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_OUTPUT)
    latencies.append((time.time() - t0) * 1000)

    pred_text  = tokenizer.decode(out[0], skip_special_tokens=True)
    label_text = ex["output"]

    preds_bool.append(parse_contradiction(pred_text))
    labels_bool.append(parse_contradiction(label_text))
    preds_type.append(parse_type(pred_text))
    labels_type.append(parse_type(label_text))

# ── Print results ─────────────────────────────────────────────────────────────
f1_binary = f1_score(labels_bool, preds_bool, zero_division=0)
f1_macro  = f1_score(labels_type, preds_type, average="macro", zero_division=0)
avg_lat   = np.mean(latencies)
p95_lat   = np.percentile(latencies, 95)

print("\n" + "="*50)
print("RESULTS")
print("="*50)
print(f"Contradiction F1 (binary):  {f1_binary:.3f}")
print(f"Type F1 (macro, 3-class):   {f1_macro:.3f}")
print(f"Avg inference latency:      {avg_lat:.1f} ms/sample")
print(f"p95 inference latency:      {p95_lat:.1f} ms/sample")
print(f"Llama-3.3-70b via Groq API: ~800-1200 ms/sample (network bound)")
print(f"Speedup vs Llama:           ~{1000/avg_lat:.0f}x")
print("="*50)

print("\nType classification report:")
print(classification_report(labels_type, preds_type, zero_division=0))

# Save results to file for resume/README
results = {
    "model": args.model,
    "train_examples": len(train_data),
    "val_examples": len(val_data),
    "epochs": args.epochs,
    "f1_binary": round(f1_binary, 3),
    "f1_macro_type": round(f1_macro, 3),
    "avg_latency_ms": round(avg_lat, 1),
    "p95_latency_ms": round(p95_lat, 1),
    "training_time_min": round(train_elapsed / 60, 1),
}
with open(os.path.join(args.output, "eval_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {args.output}/eval_results.json")
print("\nDone.")