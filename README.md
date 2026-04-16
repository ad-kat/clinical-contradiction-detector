# ClinicalContradiction

An AI system that detects cross-encounter clinical inconsistencies in longitudinal EHR data.
Built on real MIMIC-IV data using LLM-powered fact extraction and rule-based contradiction reasoning.

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

## Dataset
- Source: MIMIC-IV v3.1 (PhysioNet, credentialed access)
- 145,914 patients
- 331,793 real discharge notes
- 546,028 admissions
- 20,292,611 prescriptions
- Access requires CITI certification — completed Biomedical Research, Good Clinical Practice, and Social & Behavioral Research

## Stack
- **Data:** MIMIC-IV + MIMIC-IV-Note (PhysioNet)
- **LLM:** Groq API (Llama-3.3-70b) for structured fact extraction
- **Backend:** FastAPI + PostgreSQL
- **Infrastructure:** Docker + Render.com
- **Language:** Python

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Database connectivity check |
| GET | `/stats` | Dataset statistics |
| GET | `/patients` | List patients with discharge notes |
| GET | `/patients/{subject_id}/notes` | List all notes for a patient |
| POST | `/extract` | Extract facts from a single note |
| POST | `/detect` | Detect contradictions for a patient |

## Example Output
```json
{
  "subject_id": abcabcde,
  "notes_analyzed": 5,
  "contradictions_found": 12,
  "contradictions": [
    {
      "type": "DIAGNOSIS_DRIFT",
      "severity": "MEDIUM",
      "diagnosis": "HCV Cirrhosis",
      "first_noted_in": pqrstuvw,
      "conflict_in": lmnopqrs,
      "explanation": "Previously noted diagnosis appears resolved in a later encounter."
    },
    {
      "type": "ALLERGY_MEDICATION_CONFLICT",
      "severity": "HIGH",
      "allergy": "omeprazole",
      "allergy_noted_in": fghijklm,
      "medication": "Omeprazole 20 mg PO BID",
      "medication_prescribed_in": jklmnopq,
      "explanation": "Medication prescribed despite documented allergy."
    }
  ]
}
```

## Status
Core pipeline complete and tested on real MIMIC-IV data. Actively improving:
- [ ] Deduplication of contradictions across encounter pairs
- [ ] Chronic condition filtering (HIV, COPD, PTSD cannot truly "resolve")
- [ ] Severity scoring refinement
- [ ] Frontend dashboard