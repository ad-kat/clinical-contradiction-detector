# Project Workflow
## ClinicalContradiction — EHR Inconsistency Detection System

**Version:** 1.1  
**Author:** Adri Katyayan  
**Last Updated:** May 2026

---

## Overview

This document describes the end-to-end workflow of the ClinicalContradiction system, from raw MIMIC-IV data ingestion through contradiction detection, model training, and API serving.

---

## Phase 1 — Data Ingestion

```
MIMIC-IV v3.1 (PhysioNet, local only)
    │
    ├── discharge.csv.gz         → discharge_notes table
    ├── admissions.csv.gz        → admissions table
    ├── prescriptions.csv.gz     → prescriptions table
    └── diagnoses_icd.csv.gz     → diagnoses_icd table
                │
                ▼
        scripts/load_data.py
                │
                ▼
        PostgreSQL (local)
```

**Key constraints:**
- All ingestion is local-only
- No data is uploaded to any external service
- MIMIC-IV files are gitignored and never committed

---

## Phase 2 — Training Data Generation

```
PostgreSQL (local)
        │
        ▼
scripts/generate_training_data.py
        │
        ├── Query: patients with ≥ 2 admissions + discharge notes
        ├── Extract consecutive note pairs per patient
        ├── Run rule-based extractor (nlp/extractor.py) on each pair
        │       ├── Extract allergies, medications, diagnoses
        │       └── Detect contradictions (zero LLM calls)
        ├── Label each pair: {contradiction: True/False, type: none/allergy_medication/diagnosis_drift}
        └── Write to data/contradiction_dataset.jsonl
                │
                ▼
        8,358 labeled examples
        {none: 6000, allergy_medication: 1654, diagnosis_drift: 704}
```

---

## Phase 3 — Model Training (Knowledge Distillation)

```
data/contradiction_dataset.jsonl
        │
        ▼
scripts/deberta_classifier.py
        │
        ├── Load + parse examples
        ├── Stratified 90/10 train/val split
        ├── Tokenize (distilbert-base-uncased, MAX_LEN=128)
        ├── Compute class weights (max_count / class_count)
        ├── Fine-tune DistilBERT with WeightedTrainer
        │       ├── Task: binary (contradiction yes/no)
        │       └── Task: type (none / allergy_medication / diagnosis_drift)
        ├── Early stopping (patience=2)
        └── Save model + eval results to models/deberta-clinical/
                │
                ▼
        Binary F1: 0.731 | Type F1: 0.629 macro
        Inference: 7ms avg | ~140x faster than Groq/Llama
```

---

## Phase 4 — Production Extraction (Per Request)

```
Incoming patient note text
        │
        ▼
nlp/extractor.py (rule-based, fast path)
        │
        ├── Success → structured facts JSON
        └── Partial/failure → nlp/llm_extractor.py (Groq/Llama fallback)
                                        │
                                        ▼
                               structured facts JSON
                                        │
                                        ▼
                               nlp/detector.py
                                        │
                                ┌───────┴────────┐
                                │                │
                    ALLERGY_MEDICATION     DIAGNOSIS_DRIFT
                    _CONFLICT (HIGH)        (MEDIUM)
                                │
                                ▼
                        Contradiction objects
                        {type, severity, description, evidence}
```

---

## Phase 5 — API Serving

```
Client (browser / API consumer)
        │
        ▼
FastAPI (api/main.py)
        │
        ├── GET  /patients              → list patients + contradiction counts
        ├── GET  /patients/{id}         → single patient details
        ├── POST /analyze/{id}          → run analysis, persist results
        ├── POST /analyze/batch         → batch up to 20 patients
        ├── GET  /contradictions        → all contradictions with filters
        ├── GET  /stats                 → system-wide statistics
        ├── GET  /dashboard             → HTML dashboard
        └── GET  /health               → health check
                │
                ▼
        PostgreSQL
        (contradictions table, admissions, notes)
```

---

## Phase 6 — Dashboard

```
Browser → GET /dashboard
        │
        ▼
HTML/CSS/JS dashboard (server-rendered)
        │
        ├── Patient list panel
        │       └── Severity badges (CLEAN / 1 HIGH / N MEDIUM)
        ├── Patient detail panel
        │       ├── Contradiction cards with evidence
        │       └── Note IDs, allergy source, medication prescribed
        ├── Batch analysis mode
        └── Demo mode (synthetic patients — public deployment)
```

---

## Development Workflow

```
feature branch
    │
    ├── Write code
    ├── Run locally: docker-compose up --build
    ├── Test: pytest tests/
    │
    ▼
Pull request → main
    │
    ▼
GitHub Actions CI
    ├── pip install
    ├── pytest
    └── lint
    │
    ▼
Merge → auto-deploy to Render.com (Demo Mode)
```

---

## Planned Workflow Additions (Roadmap)

```
Phase 7 — Baseline Comparison
    scripts/baseline.py
    TF-IDF + Logistic Regression on same train/val split
    → Table 1 for paper

Phase 8 — Diagnosis Drift Augmentation
    generate_training_data.py --drift-only --n 5000
    → Increase drift examples from 704 → 2000+

Phase 9 — Human Evaluation
    Export 100 model-flagged contradictions
    Clinical expert annotation (Dr. Anupam Mishra, KGMU)
    Compute inter-rater agreement (Cohen's kappa)
    → Validation section for paper
```