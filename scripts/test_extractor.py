"""
test_extractor.py -- smoke test for LLM extraction on real notes

grabs 5 notes from the DB and runs extraction on them.
useful for checking that the Groq API key is working and the
prompt is returning sensible output.
"""

import sys
sys.path.append(".")

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from nlp.extractor import extract_all  # old regex one
# from nlp.llm_extractor import extract_all  # switch to this for LLM extraction
import os

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

with engine.connect() as conn:
    # was using noteevents before (mimic-iii), now on discharge_notes
    notes = conn.execute(text(
        "SELECT subject_id, hadm_id, note_type, text FROM discharge_notes LIMIT 5"
    )).fetchall()

for note in notes:
    print(f"\n--- Patient {note.subject_id} | Admission {note.hadm_id} | {note.note_type} ---")
    result = extract_all(note.text)
    print(f"  Allergies:   {result['allergies']}")
    print(f"  Medications: {result['medications']}")
    print(f"  Diagnoses:   {result['diagnoses']}")