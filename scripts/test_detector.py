"""
test_detector.py -- quick smoke test against real MIMIC-IV data

runs contradiction detection across all patients with discharge notes
and prints any conflicts found. not a proper unit test, just useful
for sanity-checking the detector against real notes.

NOTE: this hits the actual DB and makes Groq API calls -- can get expensive
if you have a lot of patients. run on a small subset first.

# old version used noteevents table (MIMIC-III), updated for MIMIC-IV discharge_notes
# left here in case someone is on MIMIC-III:
#
# patients = conn.execute(text(
#     "SELECT DISTINCT subject_id FROM noteevents"
# )).fetchall()
# notes = conn.execute(text(
#     "SELECT subject_id, hadm_id, category, text FROM noteevents WHERE subject_id = :sid"),
#     {"sid": subject_id}
# ).fetchall()
"""

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
        "SELECT DISTINCT subject_id FROM discharge_notes"
    )).fetchall()

    for patient in patients:
        subject_id = patient.subject_id

        notes = conn.execute(
            text("SELECT subject_id, hadm_id, note_type, charttime, text FROM discharge_notes WHERE subject_id = :sid"),
            {"sid": subject_id}
        ).fetchall()

        note_dicts = [
            {
                "subject_id": n.subject_id,
                "hadm_id": n.hadm_id,
                "note_type": n.note_type,
                "charttime": n.charttime,
                "text": n.text
            }
            for n in notes
        ]

        contradictions = detect_all_contradictions(note_dicts)
        if contradictions:
            total_contradictions += len(contradictions)
            print(f"\nPatient {subject_id} -- {len(contradictions)} contradiction(s):")
            for c in contradictions:
                score_str = f" [conf={c.get('confidence', '?')}]" if c.get('confidence') else ""
                print(f"  [{c['severity']}]{score_str} {c['type']}")
                print(f"  -> {c['explanation']}")

print(f"\nTotal contradictions detected: {total_contradictions}")