import pandas as pd
import random
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

admissions = pd.read_sql("SELECT subject_id, hadm_id FROM admissions", engine)

templates = [
    {
        "category": "Discharge summary",
        "text": lambda: f"""Admission Date: {random.randint(1,28)}/{random.randint(1,12)}/2105
Discharge Date: {random.randint(1,28)}/{random.randint(1,12)}/2105

ALLERGIES: {random.choice(['Penicillin', 'Aspirin', 'Sulfa', 'No Known Drug Allergies', 'Morphine'])}

MEDICATIONS ON ADMISSION:
{random.choice(['Metformin 500mg PO BID', 'Lisinopril 10mg PO daily', 'Atorvastatin 40mg PO nightly'])}
{random.choice(['Aspirin 81mg PO daily', 'Warfarin 5mg PO daily', 'Metoprolol 25mg PO BID'])}

DIAGNOSES:
1. {random.choice(['Type 2 Diabetes Mellitus', 'Congestive Heart Failure', 'Pneumonia', 'Acute Kidney Injury', 'COPD exacerbation'])}
2. {random.choice(['Hypertension', 'Atrial Fibrillation', 'Chronic Kidney Disease', 'Anemia', 'Sepsis'])}

HOSPITAL COURSE:
Patient was admitted with {random.choice(['chest pain', 'shortness of breath', 'altered mental status', 'fever and chills', 'acute onset confusion'])}.
{random.choice(['Patient responded well to treatment.', 'Patient required ICU level care.', 'Patient was started on IV antibiotics.'])}

DISCHARGE MEDICATIONS:
{random.choice(['Metformin 500mg PO BID', 'Lisinopril 10mg PO daily', 'Atorvastatin 40mg PO nightly'])}
{random.choice(['Aspirin 81mg PO daily', 'Penicillin VK 500mg PO QID', 'Warfarin 5mg PO daily'])}

FOLLOW UP: Patient to follow up with PCP in {random.randint(1,4)} weeks.
"""
    },
    {
        "category": "Radiology",
        "text": lambda: f"""CHEST X-RAY

INDICATION: {random.choice(['Shortness of breath', 'Chest pain', 'Fever', 'Cough'])}

FINDINGS: {random.choice([
    'No acute cardiopulmonary process. Heart size normal.',
    'Mild cardiomegaly. No pleural effusion.',
    'Bilateral lower lobe infiltrates consistent with pneumonia.',
    'Hyperinflated lungs consistent with COPD. No pneumothorax.',
])}

IMPRESSION: {random.choice([
    'No acute findings.',
    'Cardiomegaly, clinical correlation recommended.',
    'Pneumonia, recommend clinical correlation and follow up.',
    'COPD changes.',
])}
"""
    }
]

rows = []
note_id = 1
for _, row in admissions.iterrows():
    n_notes = random.randint(1, 3)
    for _ in range(n_notes):
        template = random.choice(templates)
        rows.append({
            "row_id": note_id,
            "subject_id": int(row["subject_id"]),
            "hadm_id": int(row["hadm_id"]),
            "category": template["category"],
            "text": template["text"](),
            "iserror": 0
        })
        note_id += 1

df = pd.DataFrame(rows)
df.to_sql("noteevents", engine, if_exists="replace", index=False)
print(f"Generated {len(df)} synthetic clinical notes for {len(admissions)} admissions.")
