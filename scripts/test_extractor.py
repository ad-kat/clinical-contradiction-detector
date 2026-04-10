import sys
sys.path.append(".")
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from nlp.extractor import extract_all
import os

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

with engine.connect() as conn:
    notes = conn.execute(text(
        "SELECT subject_id, hadm_id, category, text FROM noteevents LIMIT 5"
    )).fetchall()

for note in notes:
    print(f"\n--- Patient {note.subject_id} | Admission {note.hadm_id} | {note.category} ---")
    result = extract_all(note.text)
    print(f"  Allergies:   {result['allergies']}")
    print(f"  Medications: {result['medications']}")
    print(f"  Diagnoses:   {result['diagnoses']}")