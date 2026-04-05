# ClinicalContradiction

An AI system to detect cross-encounter clinical inconsistencies in longitudinal EHR data.

Built on MIMIC-IV using clinical NLP (scispaCy, BioBERT) and LLMs for semantic reasoning.

## Problem
When patients have multiple hospital admissions, their records accumulate silent contradictions —
a drug listed as tolerated in one note but flagged as an allergy in another, a diagnosis marked
resolved then reappearing, lab values implying conditions never acknowledged in clinical notes.
These contradictions cause real harm and no automated system catches them at scale.

## Stack
- Data: MIMIC-IV + MIMIC-IV-Note (PhysioNet)
- NLP: scispaCy + BioBERT
- LLM: Claude/GPT-4 API (semantic contradiction reasoning)
- Backend: FastAPI + PostgreSQL
- Deployment: Docker + Render.com
