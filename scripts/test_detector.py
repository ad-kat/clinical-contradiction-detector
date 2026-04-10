import sys
sys.path.append(".")
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from nlp.detector import detect_all_contradictions
import os
load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))
total_contradictions = 0
with engine.connect() as conn:
    patients = conn.execute(text(
        "SELECT DISTINCT subject_id FROM noteevents"
    )).fetchall()
    for patient in patients:
        subject_id = patient.subject_id
        notes = conn.execute(text(
            "SELECT subject_id, hadm_id, category, text FROM noteevents WHERE subject_id = :sid"),
            {"sid": subject_id}
        ).fetchall()
        note_dicts = [
            {"subject_id": n.subject_id, "hadm_id": n.hadm_id,
             "category": n.category, "text": n.text}
            for n in notes
        ]
        contradictions = detect_all_contradictions(note_dicts)
        if contradictions:
            total_contradictions += len(contradictions)
            print(f"\nPatient {subject_id} — {len(contradictions)} contradiction(s) found:")
            for c in contradictions:
                print(f"  [{c['severity']}] {c['type']}")
                print(f"  → {c['explanation']}")
print(f"\nTotal contradictions detected across all patients: {total_contradictions}")