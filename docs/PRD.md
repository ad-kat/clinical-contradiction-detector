# Product Requirements Document (PRD)
## ClinicalContradiction — EHR Inconsistency Detection System

**Version:** 2  
**Author:** Adri Katyayan  
**Last Updated:** May 2026  
**Status:** Active Development

---

## 1. Problem Statement

Electronic Health Records accumulate data across dozens of hospital encounters over a patient's lifetime. When a patient is admitted multiple times, clinicians document allergies, medications, and diagnoses independently per encounter — with no automated system cross-checking consistency across visits. This creates dangerous gaps:

- A patient documented with a penicillin allergy in 2021 may be prescribed amoxicillin in 2024 by a different care team with no alert
- A diagnosis of Type 2 Diabetes documented as "active" in one encounter may be contradicted as "resolved" in a subsequent note without clinical justification

These inconsistencies are a known contributor to adverse drug events and diagnostic errors, two of the leading causes of preventable patient harm in the US.

**ClinicalContradiction** addresses this by automatically detecting cross-encounter contradictions in longitudinal EHR discharge notes, surfacing them for clinical review with severity tagging.

---

## 2. Goals

| Goal | Metric | Target |
|---|---|---|
| Detect allergy-medication conflicts | Binary F1 | ≥ 0.75 |
| Classify contradiction type | Macro F1 (3-class) | ≥ 0.65 |
| Inference speed | ms/sample | < 20ms |
| Scalability | Batch size | Up to 20 patients/request |
| Compliance | Data handling | HIPAA-compliant, no PHI in transit |

---

## 3. Non-Goals

- This project does not replace clinical decision support (CDS) systems in production hospital environments
- It does not process real-time EHR streams — batch and on-demand only
- It does not provide treatment recommendations
- It does not handle imaging, lab values, or structured billing codes (ICD only as supplementary signal)

---

## 4. Users

| User | Need |
|---|---|
| Clinical researcher | Identify patterns of documentation inconsistency across a patient cohort |
| Hospital informaticist | Audit EHR data quality before a migration or integration |
| NLP researcher | Benchmark contradiction detection on MIMIC-IV |
| Recruiting reviewer | Evaluate system design, ML pipeline, and clinical NLP depth |

---

## 5. Core Features

### 5.1 Fact Extraction (Dual-Mode)
- **Rule-based (spaCy + regex):** Fast, deterministic extraction from structured note sections (ALLERGIES, MEDICATIONS ON ADMISSION, DISCHARGE DIAGNOSES). Used as teacher signal for training data generation.
- **LLM-based (Groq/Llama-3.3-70b):** Higher recall on free-text narrative sections. Used for production extraction with structured JSON output enforced via prompt engineering.

### 5.2 Contradiction Detection
- `ALLERGY_MEDICATION_CONFLICT`: Cross-references extracted allergies against prescribed medications using ingredient-level matching. Severity: HIGH.
- `DIAGNOSIS_DRIFT`: Detects conditions documented as active in one encounter and negated/resolved in a later one. Filters 80+ chronic conditions and historical-prefix phrases to reduce false positives. Severity: MEDIUM.

### 5.3 ML Classifier (Knowledge Distillation)
- Teacher: Rule-based extractor generates 8,358 labeled examples from consecutive MIMIC-IV discharge note pairs
- Student: DistilBERT (distilbert-base-uncased) fine-tuned for binary contradiction detection and 3-class type classification
- Inference: 7ms avg, ~140x faster than Groq/Llama at equivalent task

### 5.4 API
- 8 FastAPI endpoints including single-patient analysis, batch analysis (up to 20 patients), per-patient longitudinal reasoning, and severity-tagged contradiction output
- PostgreSQL schema for MIMIC-IV ingestion (admissions, prescriptions, diagnoses_icd, discharge notes)

### 5.5 Dashboard
- Patient list with contradiction severity badges
- Single-patient contradiction detail view with evidence (allergy source, medication, note IDs)
- Batch analysis mode
- Demo mode (synthetic patients, HIPAA-compliant public deployment)

---

## 6. Constraints

- MIMIC-IV data processed locally only — never uploaded to any cloud service
- Public demo uses synthetic data exclusively
- PhysioNet credentialed access required for real data pipeline
- CITI human subjects certification required for researchers accessing patient data

---

## 7. Success Metrics (Current)

| Metric | Value |
|---|---|
| Binary contradiction F1 | 0.731 |
| Type classification F1 (macro) | 0.629 |
| Avg inference latency | 7.1 ms |
| p95 inference latency | 9.0 ms |
| Speedup vs LLM baseline | ~140x |
| Training data size | 8,358 examples |
| MIMIC-IV coverage | 145,914 patients, 331,793 notes |

---

## 8. Roadmap

| Priority | Item | Status |
|---|---|---|
| P0 | Baseline comparison (TF-IDF + LR) | Planned |
| P0 | Augment diagnosis_drift training data | Planned |
| P1 | Clinical expert label validation (~100 samples) | Planned |
| P1 | Workshop paper submission (AMIA / BioNLP) | Planned |
| P2 | Multi-institution dataset (eICU, MIMIC-III) | Backlog |
| P2 | Real-time streaming support | Backlog |