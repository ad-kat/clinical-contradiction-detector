# ClinicalContradiction

An AI system that detects cross-encounter clinical inconsistencies in longitudinal EHR data.
Built on real MIMIC-IV data using LLM-powered fact extraction, rule-based contradiction reasoning,
and a fine-tuned Flan-T5 classifier trained via knowledge distillation.

**Live demo:** https://clinical-contradiction-detector.onrender.com/dashboard

## Problem
When patients have multiple hospital admissions, their records accumulate silent contradictions —
a drug prescribed despite a documented allergy, a chronic diagnosis marked resolved then
reappearing, conditions noted in one encounter never acknowledged in the next.
These inconsistencies cause real harm and no automated system catches them at scale.

## What It Does
- Extracts clinical facts (medications, allergies, diagnoses) from discharge notes using LLMs
- Detects two contradiction types across admissions:
  - `ALLERGY_MEDICATION_CONFLICT` (HIGH severity) — drug prescribed despite documented allergy
  - `DIAGNOSIS_DRIFT` (MEDIUM severity) — condition previously noted appears resolved/negated in later encounter
- Flags conflicts for clinician review with structured explanations
- **Fine-tuned Flan-T5 classifier** trained via knowledge distillation from Llama-3.3-70b on real MIMIC-IV note pairs — achieving ~15ms inference latency vs ~1000ms for the Groq API baseline (~67x speedup)

---

## Dataset & Compliance

### MIMIC-IV
This project uses **MIMIC-IV v3.1**, a large de-identified clinical database maintained by the
MIT Laboratory for Computational Physiology and distributed via
[PhysioNet](https://physionet.org/content/mimiciv/).

| Metric | Count |
|---|---|
| Patients | 145,914 |
| Discharge Notes | 331,793 |
| Admissions | 546,028 |
| Prescriptions | 20,292,611 |

### Access Requirements
MIMIC-IV is a **credentialed access** dataset. To use real data locally you must:
1. Complete CITI training in human subjects research (Data and Specimen Only Research, Biomedical Research, Good Clinical Practice, Social & Behavioral Research)
2. Create a PhysioNet account at [physionet.org](https://physionet.org)
3. Apply for credentialed access and upload your CITI certificates
4. Navigate to the MIMIC-IV and MIMIC-IV-Note project pages and sign their respective Data Use Agreements

### DUA & HIPAA Compliance
Access to MIMIC-IV is governed by the
[PhysioNet Credentialed Health Data Use Agreement](https://physionet.org/content/mimiciv/view-dua/3.1/).
By agreeing to the DUA, users commit to:
- Not attempting to identify any individual or institution in the data
- Exercising reasonable care to maintain physical and electronic security of the data
- Not sharing access to restricted data with any other party
- Using the data solely for lawful scientific research
- Openly disseminating code used to produce results (satisfied by this repository)

Although MIMIC-IV is already de-identified (names, dates, and locations have been scrubbed),
all patient data remains restricted under the DUA. This has two implications for this project:

> ⚠️ **The live dashboard cannot be deployed publicly with real MIMIC-IV data.**
> Doing so would constitute sharing credentialed access with unauthorized parties, violating
> the DUA and HIPAA regulations. The public Render deployment runs in **Demo Mode only**,
> using fully synthetic patient data. Real MIMIC-IV data is only accessible locally by
> credentialed researchers.

> ⚠️ **The distillation training dataset (`contradiction_dataset.jsonl`) is not included in this repo.**
> It contains excerpts from real MIMIC-IV discharge notes and is restricted under the same DUA.
> Credentialed researchers can regenerate it locally using `scripts/generate_training_data.py`.

---


## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  DATA LAYER (local only)                │
│          MIMIC-IV discharge notes · 331K records        │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   LLM EXTRACTION                        │
│         Groq / Llama-3.3-70b · structured JSON          │
│     medications  ·  allergies  ·  diagnoses             │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│            KNOWLEDGE DISTILLATION PIPELINE              │
│   Teacher: Llama-3.3-70b labels 2000 note pairs         │
│   Student: Flan-T5-small fine-tuned via LoRA (PEFT)     │
│   Result:  ~15ms inference  ·  F1 reported on held-out  │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                 CONTRADICTION DETECTION                 │
│      ALLERGY_MEDICATION_CONFLICT  ·  DIAGNOSIS_DRIFT    │
│                severity: HIGH / MEDIUM                  │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND                       │
│        PostgreSQL  ·  REST endpoints  ·  Docker         │
└───────────────┬─────────────────────┬───────────────────┘
                │                     │
                ▼                     ▼
   ┌────────────────────┐  ┌──────────────────────────┐
   │   LOCAL DASHBOARD  │  │    PUBLIC DEMO (Render)  │
   │  real MIMIC-IV data│  │  synthetic patients only │
   └────────────────────┘  └──────────────────────────┘
```

---

## Stack
- **Data:** MIMIC-IV + MIMIC-IV-Note (PhysioNet, credentialed)
- **LLM:** Groq API (Llama-3.3-70b) — fact extraction + distillation teacher
- **ML Model:** google/flan-t5-small fine-tuned with LoRA (PEFT, r=8) via HuggingFace Transformers
- **Backend:** FastAPI + PostgreSQL
- **Infrastructure:** Docker + Render.com (demo mode only)
- **Language:** Python

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Database connectivity check |
| GET | `/stats` | Dataset statistics |
| GET | `/patients` | List patients with discharge notes |
| GET | `/patients/{subject_id}/notes` | List all notes for a patient |
| POST | `/extract` | Extract facts from a single note |
| POST | `/detect` | Detect contradictions for a patient |
| POST | `/batch` | Batch contradiction detection (max 20 patients) |
| GET | `/dashboard` | Web dashboard UI |

---

## Local Setup (Full MIMIC-IV Pipeline)

> Requires credentialed PhysioNet access and CITI certification.

### 1. Clone the repo
```bash
git clone https://github.com/ad-kat/clinical-contradiction-detector
cd clinical-contradiction-detector
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set up PostgreSQL
```bash
sudo service postgresql start
psql -U postgres -c "CREATE DATABASE clinical_contradiction;"
psql -U postgres -c "CREATE USER adri WITH PASSWORD 'yourpassword';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE clinical_contradiction TO adri;"
```

### 4. Configure environment
```bash
cp .env.example .env
# Edit .env and fill in:
# DATABASE_URL=postgresql://adri:yourpassword@localhost/clinical_contradiction
# GROQ_API_KEY=your_groq_api_key
```

### 5. Download MIMIC-IV data
After signing the DUA, download these files from PhysioNet into `data/raw/`:

From `physionet.org/content/mimiciv/3.1/hosp/`:
- `admissions.csv.gz`
- `patients.csv.gz`
- `diagnoses_icd.csv.gz`
- `prescriptions.csv.gz`

From `physionet.org/content/mimic-iv-note/2.2/note/`:
- `discharge.csv.gz`

### 6. Load data into PostgreSQL
```bash
python scripts/load_data.py
```

### 7. Run the API
```bash
uvicorn api.main:app --reload
```

### 8. Open the dashboard
Navigate to `http://127.0.0.1:8000/dashboard`

---

## Distillation Pipeline (ML Training)

Fine-tunes a local Flan-T5 classifier using Llama-3.3-70b as a teacher on real MIMIC-IV note pairs.
All data stays local — no MIMIC data is uploaded anywhere.

### Step 1 — Generate labeled training data
Uses Llama-3.3-70b (Groq) to label real discharge note pairs extracted from MIMIC-IV.
Saves 2000 labeled examples to `data/contradiction_dataset.jsonl`.
```bash
python scripts/generate_training_data.py
```
Runtime: ~45–60 min (Groq rate-limited). Requires `data/raw/discharge.csv.gz` and `admissions.csv.gz`.

### Step 2 — Fine-tune Flan-T5 with LoRA
```bash
pip install transformers peft datasets scikit-learn accelerate torch
python scripts/finetune_flan_t5.py
```
Runtime: ~2–4 hrs on CPU (flan-t5-small). Model saved to `models/flan-t5-clinical/`.

Evaluation results (F1 + latency vs Llama baseline) saved to `models/flan-t5-clinical/eval_results.json`.

| Metric | Flan-T5-small (LoRA) | Llama-3.3-70b (Groq API) |
|---|---|---|
| Inference latency | ~15 ms | ~800–1200 ms |
| Deployment cost | Free / local | API cost per call |
| Contradiction F1 | see eval_results.json | teacher model |

---

## Public Demo (Render)

The live demo at [https://clinical-contradiction-detector.onrender.com](https://clinical-contradiction-detector.onrender.com/dashboard) runs in **Demo Mode only**.
It uses a small set of fully synthetic patients with no connection to any real clinical data.
This is intentional — see the DUA & HIPAA Compliance section above.

To run demo mode locally without any MIMIC-IV data:
```bash
DEMO_MODE=true uvicorn api.main:app --reload
```

---

## Example Output

```json
{
  "subject_id": 12345678,
  "notes_analyzed": 4,
  "contradictions_found": 3,
  "contradictions": [
    {
      "type": "DIAGNOSIS_DRIFT",
      "severity": "MEDIUM",
      "diagnosis": "Ascites",
      "first_noted_in": 11111111,
      "conflict_in": 22222222,
      "explanation": "Previously noted diagnosis appears resolved in a later encounter."
    },
    {
      "type": "ALLERGY_MEDICATION_CONFLICT",
      "severity": "HIGH",
      "allergy": "Penicillin",
      "allergy_noted_in": 11111111,
      "medication": "Amoxicillin 500mg PO TID",
      "medication_prescribed_in": 33333333,
      "explanation": "Medication prescribed despite documented allergy."
    }
  ]
}
```

---

## Status
Core pipeline complete and tested on real MIMIC-IV data. Actively improving:
- [x] Real MIMIC-IV data pipeline
- [x] LLM-powered fact extraction (Groq/Llama)
- [x] Allergy-medication conflict detection
- [x] Diagnosis drift detection
- [x] Batch analysis endpoint
- [x] Web dashboard (local + demo mode)
- [x] Chronic condition filtering refinement
- [x] Severity scoring ML layer
- [x] Render public demo deployment — live at https://clinical-contradiction-detector.onrender.com/dashboard
- [x] Knowledge distillation pipeline (Llama → Flan-T5 via LoRA/PEFT)
- [x] Local fine-tuned classifier with F1 + latency evaluation