"""
# NOTE: Deprecated — replaced by deberta_classifier.py
# flan-t5-small seq2seq approach failed (loss plateau ~1.7, F1 stuck at 0.44)
# due to model capacity limits on GTX 1650 Ti. See scripts/deberta_classifier.py.

finetune_flan_t5.py
Fine-tunes google/flan-t5-small as a 3-class sequence classifier on
MIMIC-IV derived contradiction_dataset.jsonl using HuggingFace PEFT + LoRA.

Strategy: T5ForSequenceClassification (encoder only + linear head).
Labels:
    0 = contradiction:False type:none
    1 = contradiction:True  type:allergy_medication
    2 = contradiction:True  type:diagnosis_drift

Usage:
    python scripts/finetune_flan_t5.py
    python scripts/finetune_flan_t5.py --data data/contradiction_dataset.jsonl
    python scripts/finetune_flan_t5.py --epochs 5 --batch 16

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
    T5ForSequenceClassification,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType
import torch
import torch.nn as nn

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data",   default="data/contradiction_dataset.jsonl")
parser.add_argument("--model",  default="google/flan-t5-small")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch",  type=int, default=16)
parser.add_argument("--output", default="models/flan-t5-clinical")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"Model:  {args.model}")
print(f"Data:   {args.data}")

# ── Label map ─────────────────────────────────────────────────────────────────
LABEL2ID = {
    "contradiction:False type:none":               0,
    "contradiction:True type:allergy_medication":  1,
    "contradiction:True type:diagnosis_drift":     2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 3

def output_to_label_id(output: str) -> int:
    output = output.strip()
    if output in LABEL2ID:
        return LABEL2ID[output]
    o = output.lower()
    if "allergy" in o:
        return 1
    if "drift" in o or "diagnosis" in o:
        return 2
    return 0

# ── Load JSONL ────────────────────────────────────────────────────────────────
print("\nLoading dataset...")
data = []
with open(args.data) as f:
    for line in f:
        ex = json.loads(line.strip())
        if "input" in ex and "output" in ex:
            data.append({
                "input":    ex["input"],
                "label_id": output_to_label_id(ex["output"]),
                "output":   ex["output"],
            })
        elif "text" in ex and "label" in ex:
            label = ex["label"]
            out = f"contradiction:{label['contradiction']} type:{label['type']}"
            data.append({
                "input":    f"clinical contradiction detection: {ex['text']}",
                "label_id": output_to_label_id(out),
                "output":   out,
            })
        else:
            continue

print(f"Loaded {len(data)} examples")
counts = {i: sum(1 for d in data if d["label_id"] == i) for i in range(NUM_LABELS)}
print(f"Label distribution: none={counts[0]}  allergy={counts[1]}  drift={counts[2]}")

# ── Split FIRST, then upsample train only ─────────────────────────────────────
train_data, val_data = train_test_split(
    data, test_size=0.1, random_state=42,
    stratify=[d["label_id"] for d in data]
)

allergy_pos = [ex for ex in train_data if ex["label_id"] == 1]
drift_pos   = [ex for ex in train_data if ex["label_id"] == 2]
none_pos    = [ex for ex in train_data if ex["label_id"] == 0]

target     = len(none_pos)
allergy_up = allergy_pos * (target // max(len(allergy_pos), 1))
drift_up   = drift_pos   * (target // max(len(drift_pos),   1))
train_data = none_pos + allergy_up + drift_up

print(f"\nAfter balancing:")
print(f"  none:    {len(none_pos)}")
print(f"  allergy: {len(allergy_up)}  (from {len(allergy_pos)})")
print(f"  drift:   {len(drift_up)}    (from {len(drift_pos)})")
print(f"  total train: {len(train_data)}  val: {len(val_data)}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print(f"\nLoading tokenizer: {args.model}")
tokenizer = T5Tokenizer.from_pretrained(args.model)

MAX_INPUT = 256

def clean_input(raw: str) -> str:
    raw = raw.replace("clinical contradiction detection:\n", "")
    raw = raw.replace("clinical contradiction detection: ", "")
    return f"detect contradiction: {raw.strip()}"

def tokenize_batch(batch):
    cleaned = [clean_input(x) for x in batch["input"]]
    enc = tokenizer(
        cleaned,
        max_length=MAX_INPUT,
        truncation=True,
        padding="max_length",
    )
    enc["labels"] = batch["label_id"]
    return enc

train_ds = Dataset.from_list(train_data).map(tokenize_batch, batched=True, batch_size=64)
val_ds   = Dataset.from_list(val_data).map(tokenize_batch,   batched=True, batch_size=64)

train_ds = train_ds.remove_columns(["input", "output"])
val_ds   = val_ds.remove_columns(["input", "output"])

train_ds.set_format("torch")
val_ds.set_format("torch")

print(f"\n[SANITY] First label_id: {train_ds[0]['labels']}")
print(f"[SANITY] Input ids[:5]: {train_ds[0]['input_ids'][:5].tolist()}")

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"\nLoading model: {args.model}")
model = T5ForSequenceClassification.from_pretrained(
    args.model,
    num_labels=NUM_LABELS,
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── FIX: Stronger class weights ───────────────────────────────────────────────
# Previous weights were near-uniform (0.92, 1.12, 0.98) — ineffective.
# Use hard inverse-frequency weights on ORIGINAL unbalanced counts.
orig_counts = [counts[0], counts[1], counts[2]]   # none=6000, allergy=1654, drift=704
total_orig  = sum(orig_counts)
class_weights = torch.tensor(
    [total_orig / (NUM_LABELS * max(c, 1)) for c in orig_counts],
    dtype=torch.float,
).to(DEVICE)
print(f"\n[CLASS WEIGHTS] none={class_weights[0]:.2f}  "
      f"allergy={class_weights[1]:.2f}  drift={class_weights[2]:.2f}")
# Expected: none~0.46  allergy~1.68  drift~3.95

class WeightedClassificationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss = nn.CrossEntropyLoss(weight=class_weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ── FIX: compute_metrics — handle tuple logits from T5ForSequenceClassification
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # T5ForSequenceClassification sometimes returns logits as tuple — unwrap
    if isinstance(logits, tuple):
        logits = logits[0]
    preds   = np.argmax(logits, axis=-1)
    f1_bin  = f1_score(labels > 0, preds > 0, zero_division=0)
    f1_mac  = f1_score(labels, preds, average="macro", zero_division=0)
    return {"f1_binary": round(float(f1_bin), 3), "f1_macro": round(float(f1_mac), 3)}

# ── Training args ─────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=args.output,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=args.batch,
    gradient_accumulation_steps=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    fp16=False,
    bf16=False,
    logging_steps=25,
    warmup_ratio=0.1,
    weight_decay=0.01,
    report_to="none",
    dataloader_num_workers=0,
)

trainer = WeightedClassificationTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nStarting training...")
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
preds_id, labels_id = [], []
latencies = []

for ex in val_data:
    cleaned = clean_input(ex["input"])
    inp = tokenizer(
        cleaned,
        return_tensors="pt",
        max_length=MAX_INPUT,
        truncation=True,
        padding="max_length",
    ).to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        out = model(**inp)
        logits = out.logits
        if isinstance(logits, tuple):
            logits = logits[0]
    latencies.append((time.time() - t0) * 1000)

    pred_id  = int(torch.argmax(logits, dim=-1).item())
    label_id = ex["label_id"]
    preds_id.append(pred_id)
    labels_id.append(label_id)

# Sample predictions
print("\n[DEBUG] Sample predictions (first 10):")
for i in range(min(10, len(val_data))):
    print(f"  pred:  {ID2LABEL[preds_id[i]]}")
    print(f"  label: {val_data[i]['output']}")
    print()

# ── Results ───────────────────────────────────────────────────────────────────
labels_bool = [l > 0 for l in labels_id]
preds_bool  = [p > 0 for p in preds_id]
labels_type = [ID2LABEL[l].split("type:")[1] for l in labels_id]
preds_type  = [ID2LABEL[p].split("type:")[1] for p in preds_id]

f1_binary = f1_score(labels_bool, preds_bool, zero_division=0)
f1_macro  = f1_score(labels_id, preds_id, average="macro", zero_division=0)
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

results = {
    "model": args.model,
    "architecture": "T5ForSequenceClassification",
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