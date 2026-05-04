"""
Generate labeled contradiction examples from real MIMIC-IV discharge notes.
Llama is used only as a labeler, not to generate text.
Saves to data/contradiction_dataset.jsonl
"""

import pandas as pd
import json, os, time, gzip, random
from groq import Groq, RateLimitError
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

RAW = os.path.expanduser("~/clinical-contradiction-detector/data/raw")

# ── Load MIMIC data ────────────────────────────────────────────────────────
print("Loading discharge notes...")
notes = pd.read_csv(f"{RAW}/discharge.csv.gz", compression="gzip",
                    usecols=["subject_id", "hadm_id", "text"])

print("Loading admissions...")
admissions = pd.read_csv(f"{RAW}/admissions.csv.gz", compression="gzip",
                         usecols=["subject_id", "hadm_id", "admittime"])

admissions["admittime"] = pd.to_datetime(admissions["admittime"])

# ── Find patients with 2+ admissions ──────────────────────────────────────
multi = admissions.groupby("subject_id").filter(lambda x: len(x) >= 2)
patient_ids = multi["subject_id"].unique()
print(f"Patients with 2+ admissions: {len(patient_ids)}")

# ── Extract snippet pairs from consecutive notes ───────────────────────────
LABELER_PROMPT = """You are a clinical NLP labeler. Given two short excerpts from 
consecutive hospital discharge notes for the same patient, detect contradictions.

Return ONLY valid JSON, no markdown, no preamble:
{
  "contradiction": true or false,
  "type": "allergy_medication" | "diagnosis_drift" | "none",
  "rationale": "one sentence"
}

Rules:
- allergy_medication: a drug is prescribed that the patient is allergic to
- diagnosis_drift: a prior active diagnosis is negated/resolved without explanation  
- none: no contradiction found
"""

def extract_snippet(text: str, max_chars=600) -> str:
    """Pull ALLERGIES + MEDICATIONS section from a discharge note."""
    text = str(text)
    for marker in ["ALLERGIES:", "MEDICATIONS ON ADMISSION:", "DISCHARGE MEDICATIONS:"]:
        idx = text.upper().find(marker)
        if idx != -1:
            return text[idx:idx + max_chars].strip()
    # fallback: first 600 chars
    return text[:max_chars].strip()

def label_pair(snippet1: str, snippet2: str, retries=3) -> dict:
    prompt = f"Note 1 (earlier):\n{snippet1}\n\nNote 2 (later):\n{snippet2}"
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": LABELER_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=150,
            )
            raw = resp.choices[0].message.content.strip().strip("```json").strip("```").strip()
            return json.loads(raw)
        except RateLimitError:
            wait = 15 * (attempt + 1)
            print(f"\n[rate limit] waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"\n[label error] {e}")
            return None
    return None

# ── Main loop ──────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
out_path = "data/contradiction_dataset.jsonl"

N_TARGET = 2000
written = 0

notes_by_patient = notes.groupby("subject_id")
adm_by_patient = admissions.sort_values("admittime").groupby("subject_id")

patient_list = list(patient_ids)
random.shuffle(patient_list)

with open(out_path, "w") as f:
    for pid in tqdm(patient_list, desc="Processing patients"):
        if written >= N_TARGET:
            break
        try:
            adms = adm_by_patient.get_group(pid)  # sorted by admittime
            if len(adms) < 2:
                continue
            hadm_ids = adms["hadm_id"].tolist()
            
            # get notes for first two admissions
            pat_notes = notes_by_patient.get_group(pid)
            note1 = pat_notes[pat_notes["hadm_id"] == hadm_ids[0]]["text"]
            note2 = pat_notes[pat_notes["hadm_id"] == hadm_ids[1]]["text"]
            
            if note1.empty or note2.empty:
                continue
            
            s1 = extract_snippet(note1.iloc[0])
            s2 = extract_snippet(note2.iloc[0])
            
            if len(s1) < 50 or len(s2) < 50:
                continue
            
            label = label_pair(s1, s2)
            if label is None:
                continue
            
            record = {
                "subject_id": int(pid),
                "text_pair": [s1, s2],
                # combined input for fine-tuning
                "input": f"clinical contradiction detection:\nNote 1: {s1}\nNote 2: {s2}",
                "output": f"contradiction:{label['contradiction']} type:{label['type']}",
                "label": label
            }
            f.write(json.dumps(record) + "\n")
            written += 1
            time.sleep(0.3)  # Groq rate limit
            
        except Exception as e:
            continue

print(f"\nDone. {written} examples saved to {out_path}")