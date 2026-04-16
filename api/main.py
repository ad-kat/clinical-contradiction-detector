from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import sys
import time

sys.path.append(".")
load_dotenv()

from nlp.llm_extractor import extract_all
from nlp.detector import detect_all_contradictions

app = FastAPI(
    title="ClinicalContradiction API",
    description="Detects cross-encounter clinical inconsistencies in EHR data.",
    version="1.0.0"
)
app.mount("/static", StaticFiles(directory="static"), name="static")
engine = create_engine(os.getenv("DATABASE_URL"))


# ---------- Models ----------

class NoteInput(BaseModel):
    text: str
    note_type: str = "Unknown"
    hadm_id: int | None = None
    charttime: str | None = None

class PatientRequest(BaseModel):
    subject_id: int

class BatchRequest(BaseModel):
    subject_ids: list[int]
    stop_on_first_conflict: bool = False


# ---------- Routes ----------

@app.get("/")
def root():
    return {"status": "ok", "service": "ClinicalContradiction API"}


@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.post("/extract")
def extract(note: NoteInput):
    """Extract clinical facts from a single note."""
    facts = extract_all(note.text)
    return {
        "hadm_id": note.hadm_id,
        "note_type": note.note_type,
        "charttime": note.charttime,
        "facts": facts
    }


@app.post("/detect")
def detect(req: PatientRequest):
    """Detect cross-encounter contradictions for a patient by subject_id."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT subject_id, hadm_id, note_type, charttime, text
                FROM discharge_notes
                WHERE subject_id = :sid
                ORDER BY charttime
            """),
            {"sid": req.subject_id}
        ).fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No notes found for subject_id {req.subject_id}")

    notes = [
        {
            "subject_id": r.subject_id,
            "hadm_id": r.hadm_id,
            "note_type": r.note_type,
            "charttime": r.charttime,
            "text": r.text
        }
        for r in rows
    ]

    contradictions = detect_all_contradictions(notes)

    return {
        "subject_id": req.subject_id,
        "notes_analyzed": len(notes),
        "contradictions_found": len(contradictions),
        "contradictions": contradictions
    }


@app.post("/batch")
def batch_detect(req: BatchRequest):
    """
    Run contradiction detection across multiple patients.
    Returns a summary per patient plus an aggregate report.
    Caps at 20 patients to avoid Groq rate limits.
    """
    if len(req.subject_ids) > 20:
        raise HTTPException(status_code=400, detail="Batch limit is 20 patients per request.")

    results = []
    total_contradictions = 0
    patients_with_conflicts = 0
    high_severity = 0
    medium_severity = 0

    for subject_id in req.subject_ids:
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT subject_id, hadm_id, note_type, charttime, text
                    FROM discharge_notes
                    WHERE subject_id = :sid
                    ORDER BY charttime
                """),
                {"sid": subject_id}
            ).fetchall()

        if not rows:
            results.append({
                "subject_id": subject_id,
                "status": "no_notes",
                "notes_analyzed": 0,
                "contradictions_found": 0,
                "contradictions": []
            })
            continue

        notes = [
            {
                "subject_id": r.subject_id,
                "hadm_id": r.hadm_id,
                "note_type": r.note_type,
                "charttime": r.charttime,
                "text": r.text
            }
            for r in rows
        ]

        try:
            contradictions = detect_all_contradictions(notes)
            time.sleep(5)
        except Exception as e:
            results.append({
                "subject_id": subject_id,
                "status": "error",
                "error": str(e),
                "notes_analyzed": len(notes),
                "contradictions_found": 0,
                "contradictions": []
            })
            continue

        total_contradictions += len(contradictions)
        if contradictions:
            patients_with_conflicts += 1
        high_severity += sum(1 for c in contradictions if c["severity"] == "HIGH")
        medium_severity += sum(1 for c in contradictions if c["severity"] == "MEDIUM")

        results.append({
            "subject_id": subject_id,
            "status": "ok",
            "notes_analyzed": len(notes),
            "contradictions_found": len(contradictions),
            "contradictions": contradictions
        })

        if req.stop_on_first_conflict and contradictions:
            break

    return {
        "summary": {
            "patients_requested": len(req.subject_ids),
            "patients_processed": len(results),
            "patients_with_conflicts": patients_with_conflicts,
            "total_contradictions": total_contradictions,
            "high_severity": high_severity,
            "medium_severity": medium_severity
        },
        "results": results
    }


@app.get("/patients")
def list_patients(limit: int = 100):
    """List patient subject_ids that have discharge notes."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT DISTINCT subject_id
                FROM discharge_notes
                ORDER BY subject_id
                LIMIT :limit
            """),
            {"limit": limit}
        ).fetchall()
    return {"patients": [r.subject_id for r in rows]}


@app.get("/patients/{subject_id}/notes")
def get_patient_notes(subject_id: int):
    """List all notes for a specific patient."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT note_id, hadm_id, note_type, charttime
                FROM discharge_notes
                WHERE subject_id = :sid
                ORDER BY charttime
            """),
            {"sid": subject_id}
        ).fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No notes found for subject_id {subject_id}")

    return {
        "subject_id": subject_id,
        "note_count": len(rows),
        "notes": [
            {
                "note_id": r.note_id,
                "hadm_id": r.hadm_id,
                "note_type": r.note_type,
                "charttime": r.charttime
            }
            for r in rows
        ]
    }


@app.get("/stats")
def stats():
    """Database stats."""
    with engine.connect() as conn:
        patients = conn.execute(
            text("SELECT COUNT(DISTINCT subject_id) FROM discharge_notes")).scalar()
        notes = conn.execute(
            text("SELECT COUNT(*) FROM discharge_notes")).scalar()
        admissions = conn.execute(
            text("SELECT COUNT(*) FROM admissions")).scalar()
        prescriptions = conn.execute(
            text("SELECT COUNT(*) FROM prescriptions")).scalar()
    return {
        "total_patients": patients,
        "total_discharge_notes": notes,
        "total_admissions": admissions,
        "total_prescriptions": prescriptions
    }


@app.get("/dashboard")
def dashboard():
    return FileResponse("static/dashboard.html")