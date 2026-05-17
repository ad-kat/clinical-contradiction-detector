# ClinicalContradiction

**EHR Inconsistency Detection System · MIMIC-IV**

A clinical NLP system that detects cross-encounter inconsistencies in longitudinal Electronic Health Records (EHR) data — flagging patient safety risks like allergy-medication conflicts and diagnosis drift across hospital admissions.

🔗 **[Live Demo](https://clinical-contradiction-detector.onrender.com/dashboard)** · [GitHub](https://github.com/ad-kat/clinical-contradiction-detector)

> ⚠️ The public demo runs on fully synthetic patients only, in compliance with MIMIC-IV DUA and HIPAA regulations. Real MIMIC-IV data is processed locally under PhysioNet credentialed access, obtained after completing CITI human subjects training.

---

## Problem

When patients have multiple hospital admissions, their records accumulate silent contradictions — a drug prescribed despite a documented allergy, a chronic diagnosis marked resolved then reappearing, conditions noted in one encounter never acknowledged in the next. These inconsistencies cause real harm and no automated system catches them at scale.

---

## What It Does

Given a patient's longitudinal discharge notes, ClinicalContradiction:

1. **Extracts structured facts** — allergies, medications, and diagnoses — from free-text discharge summaries
2. **Detects contradictions** across encounters:
   - `ALLERGY_MEDICATION_CONFLICT` (HIGH severity) — a drug is prescribed despite a documented allergy
   - `DIAGNOSIS_DRIFT` (MEDIUM severity) — a previously active condition is later negated or resolved without clinical justification
3. **Classifies** contradictions using a fine-tuned DistilBERT model distilled from the rule-based pipeline
4. **Surfaces** findings via a FastAPI backend and dashboard with severity tagging

---

## Results

| Task | Model | F1 | Latency (avg) | Speedup vs Llama-3.3-70b |
|---|---|---|---|---|
| Binary (contradiction yes/no) | DistilBERT | **0.731** | 7.1 ms | ~140x |
| Type classification (3-class) | DistilBERT | **0.629** macro | 6.7 ms | ~149x |

Trained on 8,358 examples derived from real MIMIC-IV v3.1 data (145,914 patients, 331,793 discharge notes, 546,028 admissions).

**Per-class breakdown (type task):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| none | 0.87 | 0.84 | 0.86 | 600 |
| allergy_medication | 0.76 | 0.82 | 0.79 | 166 |
| diagnosis_drift | 0.23 | 0.26 | 0.24 | 70 |

The low diagnosis_drift F1 reflects data scarcity (704 training examples vs 6,000 for the `none` class), not a model architecture failure. This is an active area of improvement — see [Future Work](#future-work).

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
│                   DUAL-MODE EXTRACTOR                   │
│         spaCy/regex (fast)  ·  Groq/Llama-70b (recall)  │
│     medications  ·  allergies  ·  diagnoses             │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│            KNOWLEDGE DISTILLATION PIPELINE              │
│   Teacher: Rule-based extractor labels note pairs       │
│   Student: DistilBERT fine-tuned on 8,358 examples      │
│   Result:  F1=0.731  ·  7ms inference  ·  ~140x faster  │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                 CONTRADICTION DETECTION                 │
│      ALLERGY_MEDICATION_CONFLICT  ·  DIAGNOSIS_DRIFT    │
│   DistilBERT classifier (models/deberta-clinical/)      │
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

## Model Selection: From Flan-T5 to DeBERTa to DistilBERT

The final DistilBERT classifier is the result of two failed attempts with other architectures. This section documents the reasoning and failures honestly, since the pivot decisions are as technically meaningful as the final results.

### Attempt 1 — Flan-T5-small with LoRA (deprecated)

The initial design used `google/flan-t5-small` fine-tuned with LoRA (PEFT, r=8) for seq2seq contradiction classification. This failed for fundamental architectural reasons:

- Flan-T5 is a seq2seq model — it generates output tokens rather than computing a classification logit. This makes it fragile: the model must generate exactly `"contradiction:True type:allergy_medication"` to be considered correct. Any token deviation counts as wrong.
- On a GTX 1650 Ti (4GB VRAM), fp16 caused NaN loss throughout training due to the T5 architecture's sensitivity to mixed precision. Disabling fp16 made training prohibitively slow (~4+ hours) with no convergence.
- After 4 separate debugging attempts (fixing deprecated API calls, adjusting batch sizes, rewriting compute_metrics), the model collapsed to predicting `"none"` for every input — contradiction F1: 0.000.

The script is preserved at `scripts/finetune_flan_t5.py` for reference.

### Attempt 2 — DeBERTa-v3-small (abandoned)

`microsoft/deberta-v3-small` was the natural next choice — a discriminative encoder with state-of-the-art performance on NLU tasks, including clinical NLP. It uses disentangled attention (separate position and content embeddings), which is its key architectural advantage over BERT.

The problem: disentangled attention produces large intermediate activations that overflow in fp16 on the GTX 1650 Ti. Setting `fp16=False` fixed the crash but introduced a different failure — gradients exploded to NaN at approximately epoch 0.7 in every training run, regardless of learning rate (tried 2e-5, 1e-5), warmup schedule, gradient clipping, or class weighting. The model consistently collapsed to predicting the majority class. Three separate runs produced identical failure curves.

Root cause: DeBERTa-v3's disentangled attention mechanism is numerically unstable in fp32 on older CUDA compute capabilities when combined with weighted cross-entropy loss. This is a known hardware-architecture interaction, not a hyperparameter problem.

### Final — DistilBERT (distilbert-base-uncased)

`distilbert-base-uncased` is a 66M parameter encoder distilled from BERT-base. It lacks DeBERTa's disentangled attention and is architecturally simpler — which is precisely why it works here. Standard dot-product attention is numerically stable in fp16 on any CUDA-capable GPU.

**Trade-offs vs DeBERTa:**
- DeBERTa-v3-small would likely achieve F1 ~0.77-0.80 on this task if it could be trained — it outperforms DistilBERT on most NLU benchmarks by 3-5 points
- DistilBERT achieves F1=0.731, which is lower but real and reproducible
- Inference latency is comparable: 7ms vs ~8ms for DeBERTa-small

**Trade-offs vs Groq/Llama-3.3-70b:**
- Llama is the teacher model — it has far higher recall on free-text, understands clinical context deeply, and handles edge cases DistilBERT cannot
- DistilBERT is ~140x faster at inference and has zero API cost, making it viable for batch processing thousands of patients locally
- F1=0.731 represents the compression cost: meaningful information loss relative to the LLM baseline, but acceptable for a first-pass screening tool that flags cases for human review

---

## Stack

| Layer | Technology |
|---|---|
| NLP / Extraction | spaCy, regex, Groq/Llama-3.3-70b |
| ML Classifier | DistilBERT (distilbert-base-uncased), PyTorch, HuggingFace Transformers |
| Backend | Python, FastAPI, SQLAlchemy |
| Database | PostgreSQL |
| Infrastructure | Docker, docker-compose, GitHub Actions CI |
| Deployment | Render.com |

---

## Project Structure

```
clinical-contradiction-detector/
├── api/
│   └── main.py                        # FastAPI app — 8 endpoints
├── nlp/
│   ├── detector.py                    # Contradiction detection logic
│   ├── extractor.py                   # Rule-based spaCy/regex extractor
│   └── llm_extractor.py              # Groq/Llama LLM extractor
├── scripts/
│   ├── generate_training_data.py      # Distillation dataset generator (MIMIC-IV local only)
│   ├── deberta_classifier.py          # DistilBERT fine-tuning + evaluation
│   └── finetune_flan_t5.py           # DEPRECATED — kept for reference
├── data/
│   ├── contradiction_dataset.jsonl    # 8,358 examples (gitignored — MIMIC-IV DUA)
│   └── raw/                           # discharge.csv.gz, admissions.csv.gz (local only)
├── models/
│   └── deberta-clinical/              # Saved model + eval results (gitignored)
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quickstart (Demo Mode)

```bash
git clone https://github.com/ad-kat/clinical-contradiction-detector
cd clinical-contradiction-detector
docker-compose up --build
# Visit http://localhost:8000/dashboard
```

Demo mode runs with fully synthetic patients — no MIMIC-IV data required.

---

## Local Setup (Full MIMIC-IV Pipeline)

> Requires credentialed PhysioNet access and CITI certification.

### 1. Clone and install

```bash
git clone https://github.com/ad-kat/clinical-contradiction-detector
cd clinical-contradiction-detector
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up PostgreSQL

```bash
sudo service postgresql start
psql -U postgres -c "CREATE DATABASE clinical_contradiction;"
psql -U postgres -c "CREATE USER adri WITH PASSWORD 'yourpassword';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE clinical_contradiction TO adri;"
```

### 3. Configure environment

```bash
cp .env.example .env
# Fill in:
# DATABASE_URL=postgresql://adri:yourpassword@localhost/clinical_contradiction
# GROQ_API_KEY=your_groq_api_key
```

### 4. Download MIMIC-IV data

After signing the DUA, download into `data/raw/` from PhysioNet:

- `physionet.org/content/mimiciv/3.1/hosp/` → `admissions.csv.gz`, `patients.csv.gz`, `diagnoses_icd.csv.gz`, `prescriptions.csv.gz`
- `physionet.org/content/mimic-iv-note/2.2/note/` → `discharge.csv.gz`

### 5. Load, run, open

```bash
python scripts/load_data.py
uvicorn api.main:app --reload
# Visit http://127.0.0.1:8000/dashboard
```

---

## Training the Classifier

```bash
# Generate training data (rule-based teacher labels note pairs — no Groq calls)
python scripts/generate_training_data.py --n 12000

# Train binary classifier (contradiction yes/no)
python scripts/deberta_classifier.py --task binary

# Train type classifier (none / allergy_medication / diagnosis_drift)
python scripts/deberta_classifier.py --task type
```

Results saved to `models/deberta-clinical/eval_results_binary.json` and `eval_results_type.json`.

Runtime: ~60 min per task on GTX 1650 Ti (4GB VRAM).

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Database connectivity check |
| GET | `/stats` | Dataset statistics |
| GET | `/patients` | List patients with discharge notes |
| GET | `/patients/{subject_id}/notes` | List all notes for a patient |
| POST | `/extract` | Extract facts from a single note |
| POST | `/detect` | Detect contradictions for a patient |
| POST | `/batch` | Batch contradiction detection (max 20 patients) |
| GET | `/dashboard` | Web dashboard UI |

---

## Example Output

```json
{
  "subject_id": 12345678,
  "notes_analyzed": 4,
  "contradictions_found": 2,
  "contradictions": [
    {
      "type": "ALLERGY_MEDICATION_CONFLICT",
      "severity": "HIGH",
      "allergy": "Penicillin",
      "allergy_noted_in": 11111111,
      "medication": "Amoxicillin 500mg PO TID",
      "medication_prescribed_in": 33333333,
      "explanation": "Medication prescribed despite documented allergy."
    },
    {
      "type": "DIAGNOSIS_DRIFT",
      "severity": "MEDIUM",
      "diagnosis": "Ascites",
      "first_noted_in": 11111111,
      "conflict_in": 22222222,
      "explanation": "Previously noted diagnosis appears resolved in a later encounter."
    }
  ]
}
```

---

## Dataset & Compliance

### MIMIC-IV

| Metric | Count |
|---|---|
| Patients | 145,914 |
| Discharge Notes | 331,793 |
| Admissions | 546,028 |
| Prescriptions | 20,292,611 |

### Access Requirements

MIMIC-IV is a credentialed access dataset. To use real data locally:

1. Complete CITI training (Biomedical Research, Social & Behavioral Research, Good Clinical Practice)
2. Create a PhysioNet account and apply for credentialed access
3. Sign the Data Use Agreements for MIMIC-IV and MIMIC-IV-Note

### DUA & HIPAA Compliance

> ⚠️ The live dashboard cannot be deployed publicly with real MIMIC-IV data. Doing so would constitute sharing credentialed access with unauthorized parties, violating the DUA and HIPAA. The public Render deployment runs in Demo Mode only, using fully synthetic patient data.

> ⚠️ The distillation training dataset (`contradiction_dataset.jsonl`) is not included in this repo. It contains excerpts from real MIMIC-IV discharge notes and is restricted under the same DUA. Credentialed researchers can regenerate it locally using `scripts/generate_training_data.py`.

**CITI certifications held:** Biomedical Research Investigators and Key Personnel · Social & Behavioral Research · Good Clinical Practice (all Apr 2026)

---

## Status

- [x] Real MIMIC-IV data pipeline (145K patients, 331K notes)
- [x] Dual-mode fact extraction (spaCy/regex + Groq/Llama-3.3-70b)
- [x] Allergy-medication conflict detection (HIGH severity)
- [x] Diagnosis drift detection with chronic condition filtering (MEDIUM severity)
- [x] Knowledge distillation pipeline — rule-based teacher → DistilBERT student
- [x] Binary contradiction F1: 0.731 · 7ms inference · ~140x faster than LLM baseline
- [x] Type classification F1: 0.629 macro
- [x] FastAPI backend (8 endpoints) + PostgreSQL schema
- [x] Batch analysis (up to 20 patients per request)
- [x] Docker + docker-compose deployment
- [x] GitHub Actions CI
- [x] Public demo on Render (synthetic patients, HIPAA-compliant)

---

## Future Work

**Near-term (targeting publication):**
- Baseline comparison — TF-IDF + logistic regression on the same train/val split to quantify the contribution of fine-tuning
- Diagnosis drift data augmentation — increase from 704 to 2,000+ training examples via targeted MIMIC-IV extraction and text augmentation; current F1=0.24 on this class is the system's main weakness
- Clinical expert label validation — manual review of ~100 model-flagged contradictions by a clinician to establish label validity and compute inter-rater agreement (Cohen's kappa); planned with Dr. Anupam Mishra (Dept. of Otolaryngology & Head and Neck Surgery, KGMU)
- Target venue: AMIA Annual Symposium (clinical NLP track) or BioNLP Workshop @ ACL

**Longer-term:**
- Multi-institution generalization — extend beyond MIMIC-IV (single institution, Beth Israel Deaconess Medical Center) to eICU or MIMIC-III for cross-site validation
- Additional contradiction types — medication dosage conflicts, lab value contradictions, procedure duplication
- Real-time EHR integration — streaming support for live contradiction flagging at point of care
- DeBERTa-v3 on adequate hardware — the architecture is better suited to this task; training failed on GTX 1650 Ti due to fp16 instability, but would likely achieve F1 ~0.77-0.80 on a GPU with ≥8GB VRAM

---

## Author

**Adri Katyayan** · MS Computer Science, Stony Brook University  
[LinkedIn](https://www.linkedin.com/in/adri-katyayan-21a0b2222/) · [GitHub](https://github.com/ad-kat) · adri.katyayan@stonybrook.edu