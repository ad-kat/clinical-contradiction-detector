"""
generate_training_data.py

Generates labeled contradiction training pairs from real MIMIC-IV discharge notes.
Uses the existing rule-based extractor (nlp/extractor.py) -- NO Groq, NO API calls.

Runtime: ~5-15 min for 2000 examples.

Usage:
    python scripts/generate_training_data.py
    python scripts/generate_training_data.py --n 2000 --out data/contradiction_dataset.jsonl

Requires:
    data/raw/discharge.csv.gz
    data/raw/admissions.csv.gz
"""

import sys
import os
import json
import argparse
import random
import re

import pandas as pd
from tqdm import tqdm

# ── make sure project root is on path so nlp/ imports work ──────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nlp.extractor import extract_allergies, extract_medications, extract_diagnoses

# ── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n",   type=int, default=2000, help="Target number of examples")
parser.add_argument("--raw", default=os.path.expanduser("~/clinical-contradiction-detector/data/raw"))
parser.add_argument("--out", default="data/contradiction_dataset.jsonl")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
RESOLVED_KEYWORDS = [
    "resolved", "no longer", "ruled out", "negative for",
    "denies", "no evidence of", "without", "absent", "cleared",
]

HISTORICAL_PREFIXES = (
    "hx of", "history of", "h/o", "past history of",
    "prior history of", "previous history of", "pmh of",
)

CHRONIC_CONDITIONS = {
    "diabetes", "hypertension", "heart failure", "copd", "asthma",
    "atrial fibrillation", "chronic kidney disease", "ckd", "cirrhosis",
    "hiv", "cancer", "carcinoma", "lymphoma", "epilepsy", "dementia",
    "parkinson", "multiple sclerosis", "bipolar", "schizophrenia",
    "rheumatoid arthritis", "lupus", "crohn", "ulcerative colitis",
    "osteoarthritis", "osteoporosis", "hypothyroidism", "hyperthyroidism",
}

HIGH_RISK_DRUG_CLASSES = {
    "penicillin": ["amoxicillin", "ampicillin", "piperacillin", "nafcillin",
                   "augmentin", "amoxicillin-clavulanate"],
    "cephalosporin": ["cephalexin", "cefazolin", "ceftriaxone", "cefepime", "cefdinir"],
    "sulfa": ["trimethoprim-sulfamethoxazole", "sulfamethoxazole", "bactrim"],
    "nsaid": ["ibuprofen", "naproxen", "ketorolac", "meloxicam", "aspirin"],
    "opioid": ["morphine", "oxycodone", "hydrocodone", "fentanyl", "codeine", "tramadol"],
}

# keyword filter — only load notes with clinical substance
FILTER_KEYWORDS = ["allerg", "medication", "diagnos", "prescribed", "penicillin",
                   "resolved", "no longer", "ruled out"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return text.lower().strip()

def is_historical(diagnosis: str) -> bool:
    d = normalize(diagnosis)
    return any(d.startswith(p) for p in HISTORICAL_PREFIXES)

def is_chronic(diagnosis: str) -> bool:
    d = normalize(diagnosis)
    return any(c in d for c in CHRONIC_CONDITIONS)

def allergy_class_conflict(allergy: str, medication: str) -> bool:
    a = normalize(allergy).split()[0]
    m = normalize(medication).split()[0]
    for cls, members in HIGH_RISK_DRUG_CLASSES.items():
        if a == cls or a in cls:
            if any(m in mem or mem.startswith(m) for mem in members):
                return True
    return False

def extract_snippet(text: str, max_chars: int = 500) -> str:
    """Pull first allergy/medication section found, fallback to first 500 chars."""
    for marker in ["ALLERGIES:", "MEDICATIONS ON ADMISSION:", "DISCHARGE MEDICATIONS:", "DIAGNOSES:"]:
        idx = text.upper().find(marker)
        if idx != -1:
            return text[idx: idx + max_chars].strip()
    return text[:max_chars].strip()

def has_keywords(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in FILTER_KEYWORDS)

# ── Label a note pair using rule-based logic ──────────────────────────────────

def label_pair(text1: str, text2: str) -> dict:
    """
    Returns label dict:
      contradiction: bool
      type: "allergy_medication" | "diagnosis_drift" | "none"
      rationale: str
    """
    allergies1  = extract_allergies(text1)
    meds2       = extract_medications(text2)
    diagnoses1  = extract_diagnoses(text1)
    text2_lower = text2.lower()

    # ── Check allergy-medication conflict ─────────────────────────────────────
    for allergy in allergies1:
        a_key = normalize(allergy).split()[0]
        for med in meds2:
            m_key = normalize(med).split()[0]
            exact = a_key in m_key or m_key in a_key
            cls   = allergy_class_conflict(allergy, med)
            if exact or cls:
                return {
                    "contradiction": True,
                    "type": "allergy_medication",
                    "rationale": (
                        f"'{med}' prescribed in note 2 despite allergy to '{allergy}' in note 1."
                        if exact else
                        f"'{med}' is in same drug class as documented allergy '{allergy}'."
                    ),
                }

    # ── Check diagnosis drift ─────────────────────────────────────────────────
    for diag in diagnoses1:
        if is_historical(diag) or is_chronic(diag):
            continue
        diag_key = normalize(diag)
        if len(diag_key) < 4:
            continue
        if diag_key in text2_lower:
            for kw in RESOLVED_KEYWORDS:
                if kw in text2_lower:
                    # check keyword is near diagnosis mention
                    idx_diag = text2_lower.find(diag_key)
                    idx_kw   = text2_lower.find(kw)
                    if abs(idx_diag - idx_kw) < 300:
                        return {
                            "contradiction": True,
                            "type": "diagnosis_drift",
                            "rationale": (
                                f"'{diag}' noted in note 1 appears '{kw}' in note 2."
                            ),
                        }

    # ── No contradiction ──────────────────────────────────────────────────────
    return {
        "contradiction": False,
        "type": "none",
        "rationale": "No allergy conflict or diagnosis drift detected between notes.",
    }

# ── Load MIMIC data ───────────────────────────────────────────────────────────
print("Loading admissions...")
admissions = pd.read_csv(
    os.path.join(args.raw, "admissions.csv.gz"),
    compression="gzip",
    usecols=["subject_id", "hadm_id", "admittime"],
)
admissions["admittime"] = pd.to_datetime(admissions["admittime"])
admissions = admissions.sort_values(["subject_id", "admittime"])

print("Loading discharge notes (this may take ~1-2 min)...")
notes = pd.read_csv(
    os.path.join(args.raw, "discharge.csv.gz"),
    compression="gzip",
    usecols=["subject_id", "hadm_id", "text"],
)

# pre-filter: only notes with relevant clinical content
print("Filtering notes...")
mask  = notes["text"].apply(has_keywords)
notes = notes[mask].copy()
print(f"Notes after filter: {len(notes):,} / {mask.shape[0]:,}")

# ── Build patient → sorted note list index ────────────────────────────────────
print("Indexing patients...")
notes_idx = notes.groupby("subject_id")
adm_idx   = admissions.groupby("subject_id")

# only patients with 2+ admissions AND notes
multi_adm = admissions.groupby("subject_id").filter(lambda x: len(x) >= 2)
patient_ids = [
    pid for pid in multi_adm["subject_id"].unique()
    if pid in notes_idx.groups
]
random.shuffle(patient_ids)
print(f"Eligible patients: {len(patient_ids):,}")

# ── Main loop ─────────────────────────────────────────────────────────────────
written    = 0
n_positive = 0  # track class balance
n_negative = 0

TARGET_POS = args.n // 2   # aim for ~50/50 balance
TARGET_NEG = args.n // 2

with open(args.out, "w") as f:
    for pid in tqdm(patient_ids, desc="Processing patients"):
        if written >= args.n:
            break

        try:
            pat_adms  = adm_idx.get_group(pid).sort_values("admittime")
            pat_notes = notes_idx.get_group(pid)
            hadm_ids  = pat_adms["hadm_id"].tolist()

            # slide a window over consecutive admission pairs
            for i in range(len(hadm_ids) - 1):
                if written >= args.n:
                    break

                h1 = hadm_ids[i]
                h2 = hadm_ids[i + 1]

                rows1 = pat_notes[pat_notes["hadm_id"] == h1]["text"]
                rows2 = pat_notes[pat_notes["hadm_id"] == h2]["text"]

                if rows1.empty or rows2.empty:
                    continue

                text1 = str(rows1.iloc[0])
                text2 = str(rows2.iloc[0])

                if len(text1) < 100 or len(text2) < 100:
                    continue

                label = label_pair(text1, text2)

                # balance classes
                is_pos = label["contradiction"]
                if is_pos and n_positive >= TARGET_POS:
                    continue
                if not is_pos and n_negative >= TARGET_NEG:
                    continue

                snippet1 = extract_snippet(text1)
                snippet2 = extract_snippet(text2)

                record = {
                    "subject_id": int(pid),
                    "hadm_pair":  [int(h1), int(h2)],
                    "input":  (
                        f"clinical contradiction detection:\n"
                        f"Note 1: {snippet1}\n"
                        f"Note 2: {snippet2}"
                    ),
                    "output": (
                        f"contradiction:{label['contradiction']} "
                        f"type:{label['type']}"
                    ),
                    "label": label,
                }

                f.write(json.dumps(record) + "\n")
                written += 1
                if is_pos:
                    n_positive += 1
                else:
                    n_negative += 1

        except Exception:
            continue

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nDone.")
print(f"Total examples : {written}")
print(f"Positive (contradiction=True)  : {n_positive}")
print(f"Negative (contradiction=False) : {n_negative}")
print(f"Saved to: {args.out}")