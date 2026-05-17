# Technical Requirements Document (TRD)
## ClinicalContradiction — EHR Inconsistency Detection System

**Version:** 2  
**Author:** Adri Katyayan  
**Last Updated:** May 2026

---

## 1. System Overview

ClinicalContradiction is a clinical NLP system built on a three-layer architecture: a dual-mode fact extraction pipeline, a rule-based contradiction detection engine, and a fine-tuned DistilBERT classifier trained via knowledge distillation. The system ingests MIMIC-IV discharge notes, extracts structured medical facts, and surfaces cross-encounter inconsistencies via a FastAPI backend and PostgreSQL database.

---

## 2. Technology Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.12 |
| NLP — rule-based | spaCy | ≥ 3.7 |
| NLP — LLM | Groq API / Llama-3.3-70b | — |
| ML framework | PyTorch | ≥ 2.0 |
| Transformers | HuggingFace Transformers | ≥ 4.40 |
| Classifier model | distilbert-base-uncased | — |
| API framework | FastAPI | ≥ 0.110 |
| ORM | SQLAlchemy | ≥ 2.0 |
| Database | PostgreSQL | 15 |
| Containerization | Docker, docker-compose | — |
| CI/CD | GitHub Actions | — |
| Deployment | Render.com | — |

---

## 3. Data Pipeline

### 3.1 Ingestion (`scripts/load_data.py`)
- Source: MIMIC-IV v3.1 CSV files (`discharge.csv.gz`, `admissions.csv.gz`, `prescriptions.csv.gz`, `diagnoses_icd.csv.gz`)
- Loaded into PostgreSQL with the following tables: `admissions`, `discharge_notes`, `prescriptions`, `diagnoses_icd`
- All ingestion runs locally under PhysioNet credentialed access — no data leaves the local environment

### 3.2 Training Data Generation (`scripts/generate_training_data.py`)
- Extracts consecutive discharge note pairs per patient (minimum 2 admissions)
- Labels using the rule-based extractor (zero LLM calls — deterministic, reproducible)
- Output: `data/contradiction_dataset.jsonl` — 8,358 examples
- Label distribution: `{none: 6000, allergy_medication: 1654, diagnosis_drift: 704}`

---

## 4. Extraction Pipeline

### 4.1 Rule-Based Extractor (`nlp/extractor.py`)
Parses structured sections of discharge notes using regex and spaCy:
- `ALLERGIES:` section → allergy list
- `MEDICATIONS ON ADMISSION:` section → medication list
- `DISCHARGE DIAGNOSES:` section → diagnosis list

Returns structured JSON: `{allergies: [...], medications: [...], diagnoses: [...]}`.

### 4.2 LLM Extractor (`nlp/llm_extractor.py`)
- Model: Groq API → `llama-3.3-70b-versatile`
- Prompt: structured JSON output enforced via system prompt + post-processing validation
- Used for production extraction on free-text narrative sections where regex recall is insufficient
- Fallback: rule-based extractor on API failure or timeout

---

## 5. Contradiction Detection Engine (`nlp/detector.py`)

### 5.1 ALLERGY_MEDICATION_CONFLICT
```
for each admission pair (earlier, later):
    allergies = extract(earlier.note).allergies
    medications = extract(later.note).medications
    for each medication in medications:
        if ingredient_match(medication, allergies):
            flag(HIGH, medication, allergy, note_ids)
```
Ingredient matching uses a curated mapping of brand → generic drug names plus substring normalization.

### 5.2 DIAGNOSIS_DRIFT
```
for each admission pair (earlier, later):
    active_dx = extract(earlier.note).diagnoses — filter(chronic_conditions)
    later_text = later.note.full_text
    for each dx in active_dx:
        if negation_pattern_match(dx, later_text):
            flag(MEDIUM, dx, note_ids)
```
- Chronic condition filter: 80+ conditions (hypertension, diabetes, COPD, etc.) excluded from drift detection
- Historical prefix suppression: phrases like "history of", "h/o", "prior" suppress false positive flags

---

## 6. ML Classifier (`scripts/deberta_classifier.py`)

### 6.1 Architecture
- Base model: `distilbert-base-uncased` (66M parameters)
- Task heads: binary sequence classification (2-class) and type classification (3-class)
- Training: WeightedTrainer with class-weighted cross-entropy loss
- Class weights: `max_count / class_count` per label

### 6.2 Training Configuration

| Parameter | Value |
|---|---|
| Max sequence length | 128 tokens |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| fp16 | True (CUDA) |
| Early stopping patience | 2 epochs |
| Train/val split | 90/10, stratified |

### 6.3 Results

| Task | F1 | Precision | Recall | Latency (avg) |
|---|---|---|---|---|
| Binary (contradiction) | 0.731 | 0.78 | 0.69 | 7.1 ms |
| Type — none | 0.86 | 0.87 | 0.84 | — |
| Type — allergy_medication | 0.79 | 0.76 | 0.82 | — |
| Type — diagnosis_drift | 0.24 | 0.23 | 0.26 | 6.7 ms |
| Type — macro avg | 0.629 | 0.62 | 0.64 | — |

---

## 7. API Specification

Base URL: `https://clinical-contradiction-detector.onrender.com`

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/dashboard` | Frontend dashboard (HTML) |
| GET | `/patients` | List all patients with contradiction counts |
| GET | `/patients/{patient_id}` | Single patient details |
| POST | `/analyze/{patient_id}` | Run contradiction analysis for one patient |
| POST | `/analyze/batch` | Batch analysis (up to 20 patients) |
| GET | `/contradictions` | All detected contradictions with filters |
| GET | `/stats` | System-wide statistics |

---

## 8. Database Schema

```sql
-- Admissions
CREATE TABLE admissions (
    hadm_id     INTEGER PRIMARY KEY,
    subject_id  INTEGER NOT NULL,
    admittime   TIMESTAMP,
    dischtime   TIMESTAMP,
    admission_type TEXT
);

-- Discharge notes
CREATE TABLE discharge_notes (
    note_id     SERIAL PRIMARY KEY,
    hadm_id     INTEGER REFERENCES admissions(hadm_id),
    subject_id  INTEGER,
    charttime   TIMESTAMP,
    text        TEXT
);

-- Prescriptions
CREATE TABLE prescriptions (
    row_id      SERIAL PRIMARY KEY,
    hadm_id     INTEGER REFERENCES admissions(hadm_id),
    subject_id  INTEGER,
    drug        TEXT,
    dose_val_rx TEXT
);

-- Diagnoses (ICD)
CREATE TABLE diagnoses_icd (
    row_id      SERIAL PRIMARY KEY,
    hadm_id     INTEGER REFERENCES admissions(hadm_id),
    subject_id  INTEGER,
    icd_code    TEXT,
    icd_version INTEGER
);

-- Detected contradictions
CREATE TABLE contradictions (
    id              SERIAL PRIMARY KEY,
    subject_id      INTEGER,
    hadm_id_earlier INTEGER,
    hadm_id_later   INTEGER,
    contradiction_type TEXT,  -- allergy_medication | diagnosis_drift
    severity        TEXT,     -- HIGH | MEDIUM
    description     TEXT,
    evidence        JSONB,
    detected_at     TIMESTAMP DEFAULT NOW()
);
```

---

## 9. Security & Compliance

- No PHI stored in application database beyond what is required for local research use
- `.gitignore` excludes all MIMIC-IV raw data, processed datasets, and trained models
- Public deployment (Render.com) runs in Demo Mode exclusively — synthetic data only
- CITI certification obtained: Data and Specimens Only Research, Biomedical Research, Social & Behavioral Research, Good Clinical Practice (Apr 2026)
- PhysioNet credentialed access obtained via standard academic review process

---

## 10. Infrastructure

```
Local (WSL2 / Linux):
  - PostgreSQL 15 (local instance)
  - NVIDIA GTX 1650 Ti (4GB VRAM) — model training
  - MIMIC-IV raw data (local only, gitignored)

Docker (local + cloud):
  - api service: FastAPI app
  - db service: PostgreSQL
  - health-gated startup via depends_on: condition: service_healthy

CI/CD:
  - GitHub Actions: lint + pytest on push to main

Production (Render.com):
  - Demo Mode only
  - Synthetic patient data
```