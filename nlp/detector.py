# from nlp.extractor import extract_all
from nlp.llm_extractor import extract_all

def normalize(text: str) -> str:
    return text.lower().strip()

def extract_all_notes(notes: list[dict]) -> list[dict]:
    """Call LLM once per note, cache results."""
    return [
        {**note, "_facts": extract_all(note["text"])}
        for note in notes
    ]

def detect_allergy_medication_conflict(extracted: list[dict]) -> list[dict]:
    allergy_sources = {}
    for note in extracted:
        for allergy in note["_facts"]["allergies"]:
            key = normalize(allergy)
            if key not in allergy_sources:
                allergy_sources[key] = {"hadm_id": note["hadm_id"], "raw": allergy}

    seen = set()
    conflicts = []
    for note in extracted:
        for med in note["_facts"]["medications"]:
            med_key = normalize(med.split()[0])
            for allergy_key, source in allergy_sources.items():
                if allergy_key in med_key or med_key in allergy_key:
                    dedup_key = (allergy_key, med_key)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    conflicts.append({
                        "type": "ALLERGY_MEDICATION_CONFLICT",
                        "severity": "HIGH",
                        "allergy": source["raw"],
                        "allergy_noted_in": source["hadm_id"],
                        "medication": med,
                        "medication_prescribed_in": note["hadm_id"],
                        "category": note.get("category"),
                        "explanation": f"'{med}' was prescribed but '{source['raw']}' is listed as an allergy."
                    })
    return conflicts

def detect_diagnosis_drift(extracted: list[dict]) -> list[dict]:
    resolved_keywords = ["resolved", "no longer", "ruled out", "negative for"]
    seen_diagnoses = {}

    for note in sorted(extracted, key=lambda x: x["hadm_id"]):
        for diagnosis in note["_facts"]["diagnoses"]:
            key = normalize(diagnosis)
            if key not in seen_diagnoses:
                seen_diagnoses[key] = {"hadm_id": note["hadm_id"], "raw": diagnosis}

    seen = set()
    drift = []
    for note in extracted:
        text_lower = note["text"].lower()
        for key, source in seen_diagnoses.items():
            for keyword in resolved_keywords:
                if keyword in text_lower and key in text_lower:
                    if note["hadm_id"] != source["hadm_id"]:
                        dedup_key = (key, keyword, note["hadm_id"])
                        if dedup_key in seen:
                            continue
                        seen.add(dedup_key)
                        drift.append({
                            "type": "DIAGNOSIS_DRIFT",
                            "severity": "MEDIUM",
                            "diagnosis": source["raw"],
                            "first_noted_in": source["hadm_id"],
                            "conflict_in": note["hadm_id"],
                            "explanation": f"'{source['raw']}' was previously noted but appears '{keyword}' in a later encounter."
                        })
    return drift

def detect_all_contradictions(notes: list[dict]) -> list[dict]:
    extracted = extract_all_notes(notes)
    contradictions = []
    contradictions += detect_allergy_medication_conflict(extracted)
    contradictions += detect_diagnosis_drift(extracted)
    return contradictions