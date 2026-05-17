"""
deberta_classifier.py
Fine-tunes distilbert-base-uncased on contradiction_dataset.jsonl
for clinical contradiction detection (sequence classification).

NOTE: Switched from microsoft/deberta-v3-small — DeBERTa disentangled attention
produces NaN overflow on GTX 1650 Ti regardless of fp16/fp32 settings.
DistilBERT is stable, fast (~15-20 min), and still produces strong F1.

Usage:
    python scripts/deberta_classifier.py --task binary
    python scripts/deberta_classifier.py --task type

Requirements:
    pip install transformers datasets scikit-learn accelerate torch sentencepiece protobuf
"""

import json
import time
import argparse
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
import torch
import torch.nn as nn

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data",   default="data/contradiction_dataset.jsonl")
parser.add_argument("--model",  default="distilbert-base-uncased")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch",  type=int, default=16)
parser.add_argument("--output", default="models/deberta-clinical")
parser.add_argument("--task",   default="binary",
                    choices=["binary", "type"],
                    help="binary = contradiction yes/no | type = 3-class type")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"Model:  {args.model}")
print(f"Data:   {args.data}")
print(f"Task:   {args.task}")

# ── Label maps ────────────────────────────────────────────────────────────────
BINARY_LABELS = {"False": 0, "True": 1}
TYPE_LABELS = {
    "none": 0,
    "allergy_medication": 1,
    "diagnosis_drift": 2,
}

def parse_output(output_str: str):
    contradiction = "False"
    ctype = "none"
    try:
        contradiction = output_str.split("contradiction:")[1].split()[0].strip()
    except IndexError:
        pass
    try:
        ctype = output_str.split("type:")[1].strip().split()[0]
    except IndexError:
        pass
    return contradiction, ctype

# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading dataset...")
data = []
with open(args.data) as f:
    for line in f:
        ex = json.loads(line.strip())
        if "input" not in ex or "output" not in ex:
            continue
        contradiction, ctype = parse_output(ex["output"])
        if args.task == "binary":
            label = BINARY_LABELS.get(contradiction, 0)
        else:
            label = TYPE_LABELS.get(ctype, 0)
        data.append({"text": ex["input"], "label": label})

print(f"Loaded {len(data)} examples")

label_counts = Counter(d["label"] for d in data)
print(f"Label distribution: {dict(label_counts)}")

# ── Class weights ─────────────────────────────────────────────────────────────
num_labels = 2 if args.task == "binary" else 3
max_count = max(label_counts.values())
# balanced: max_count / class_count (stronger than linear)
class_weights = torch.tensor(
    [max_count / label_counts[i] for i in range(num_labels)],
    dtype=torch.float32,
)
print(f"Class weights: {class_weights.tolist()}")

# ── Train/val split ───────────────────────────────────────────────────────────
train_data, val_data = train_test_split(
    data, test_size=0.1, random_state=42,
    stratify=[d["label"] for d in data]
)
print(f"Train: {len(train_data)}  Val: {len(val_data)}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print(f"\nLoading tokenizer: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model)

MAX_LEN = 128

def tokenize(batch):
    return tokenizer(
        batch["text"],
        max_length=MAX_LEN,
        truncation=True,
        padding=False,
    )

train_ds = Dataset.from_list(train_data).map(tokenize, batched=True, batch_size=64)
val_ds   = Dataset.from_list(val_data).map(tokenize,   batched=True, batch_size=64)

train_ds = train_ds.rename_column("label", "labels")
val_ds   = val_ds.rename_column("label", "labels")
train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_ds.set_format("torch",   columns=["input_ids", "attention_mask", "labels"])

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"\nLoading model: {args.model} ({num_labels} labels)")
model = AutoModelForSequenceClassification.from_pretrained(
    args.model,
    num_labels=num_labels,
)
model.to(DEVICE)

# ── Weighted Trainer ──────────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # match dtype+device of logits — avoids Half/Float mismatch
        weights = class_weights.to(dtype=logits.dtype, device=logits.device)
        loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ── Compute metrics ───────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    if args.task == "binary":
        f1 = f1_score(labels, preds, average="binary", zero_division=0)
        return {"f1_binary": round(f1, 3)}
    else:
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        return {"f1_macro": round(f1, 3)}

# ── Training args ─────────────────────────────────────────────────────────────
use_fp16 = DEVICE == "cuda"   # DistilBERT handles fp16 fine

training_args = TrainingArguments(
    output_dir=args.output,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=args.batch,
    gradient_accumulation_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_binary" if args.task == "binary" else "f1_macro",
    greater_is_better=True,
    fp16=use_fp16,
    logging_steps=50,
    warmup_ratio=0.1,           # 10% of steps — more stable than fixed warmup_steps
    weight_decay=0.01,
    learning_rate=2e-5,
    max_grad_norm=1.0,
    report_to="none",
    dataloader_num_workers=0,   # WSL2 safe
)

data_collator = DataCollatorWithPadding(tokenizer)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nStarting training...")
print(f"MAX_LEN={MAX_LEN}, batch={args.batch}, lr=2e-5, fp16={use_fp16}, weighted loss")
print("Expected: ~15-20 min on GTX 1650 Ti\n")

train_start = time.time()
trainer.train()
train_elapsed = time.time() - train_start
print(f"\nTraining done in {train_elapsed/60:.1f} min")

trainer.save_model(args.output)
tokenizer.save_pretrained(args.output)
print(f"Model saved to {args.output}/")

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\nEvaluating on validation set...")

model.eval()
preds_all, labels_all = [], []
latencies = []

for ex in val_data:
    inp = tokenizer(
        ex["text"],
        return_tensors="pt",
        max_length=MAX_LEN,
        truncation=True,
    ).to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        out = model(**inp)
    latencies.append((time.time() - t0) * 1000)

    pred = int(torch.argmax(out.logits, dim=-1).cpu())
    preds_all.append(pred)
    labels_all.append(ex["label"])

# ── Results ───────────────────────────────────────────────────────────────────
avg_lat = np.mean(latencies)
p95_lat = np.percentile(latencies, 95)

if args.task == "binary":
    f1 = f1_score(labels_all, preds_all, average="binary", zero_division=0)
    label_names = ["no_contradiction", "contradiction"]
else:
    f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    label_names = ["none", "allergy_medication", "diagnosis_drift"]

print("\n" + "="*55)
print("RESULTS")
print("="*55)
print(f"Task:                       {args.task}")
print(f"F1 score:                   {f1:.3f}")
print(f"Avg inference latency:      {avg_lat:.1f} ms/sample")
print(f"p95 inference latency:      {p95_lat:.1f} ms/sample")
print(f"Llama-3.3-70b via Groq:     ~800-1200 ms/sample")
print(f"Speedup vs Llama:           ~{1000/avg_lat:.0f}x")
print("="*55)

print("\nClassification report:")
print(classification_report(labels_all, preds_all,
                             target_names=label_names, zero_division=0))

results = {
    "model": args.model,
    "task": args.task,
    "train_examples": len(train_data),
    "val_examples": len(val_data),
    "epochs": args.epochs,
    "f1": round(f1, 3),
    "avg_latency_ms": round(avg_lat, 1),
    "p95_latency_ms": round(p95_lat, 1),
    "training_time_min": round(train_elapsed / 60, 1),
}
out_path = os.path.join(args.output, f"eval_results_{args.task}.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {out_path}")
print("\nDone.")