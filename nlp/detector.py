from nlp.llm_extractor import extract_all

# Chronic conditions that cannot truly resolve — flag differently
CHRONIC_CONDITIONS = {
    "hiv", "copd", "bipolar", "ptsd", "diabetes", "hypertension",
    "heart failure", "chronic kidney disease", "ckd", "hepatitis c",
    "hcv", "asthma", "epilepsy", "schizophrenia", "multiple sclerosis"
}

# Prefixes that indicate already-historical diagnoses — skip these
HISTORICAL_PREFIXES = (
    "hx of", "history of", "h/o", "past history of",
    "prior history of", "previous history of", "pmh of"
)

def normalize(text: str) -> str:
    return text.lower().strip()

def is_chronic(diagnosis: str) -> bool:
    d = normalize(diagnosis)
    return any(c in d for c in CHRONIC_CONDITIONS)

def is_historical(diagnosis: str) -> bool:
    d = normalize(diagnosis)
    return any(d.startswith(prefix) for prefix in HISTORICAL_PREFIXES)

def normalize_diagnosis(diagnosis: str) -> str:
    """Strip historical prefixes for fuzzy matching."""
    d = normalize(diagnosis)
    for prefix in HISTORICAL_PREFIXES:
        if d.startswith(prefix):
            d = d[len(prefix):].strip()
    return d

def extract_all_notes(notes: list[dict]) -> list[dict]:
    """Call LLM once per note, cache results."""
    return [
        {**note, "_facts": extract_all(note["text"])}
        for note in notes
    ]

def detect_allergy_medication_conflict(extracted: list[dict]) -> list[dict]:
    sorted_notes = sorted(extracted, key=lambda x: x.get("charttime") or "")

    allergy_sources = {}
    for note in sorted_notes:
        for allergy in note["_facts"]["allergies"]:
            key = normalize(allergy)
            if key not in allergy_sources:
                allergy_sources[key] = {"hadm_id": note["hadm_id"], "raw": allergy}

    seen = set()
    conflicts = []
    for note in sorted_notes:
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
                        "explanation": f"'{med}' was prescribed but '{source['raw']}' is listed as an allergy."
                    })
    return conflicts

def detect_diagnosis_drift(extracted: list[dict]) -> list[dict]:
    resolved_keywords = ["resolved", "no longer", "ruled out", "negative for"]

    sorted_notes = sorted(extracted, key=lambda x: x.get("charttime") or "")

    # Build first-seen diagnosis map using normalized names
    seen_diagnoses = {}
    for note in sorted_notes:
        for diagnosis in note["_facts"]["diagnoses"]:
            # Skip historical and chronic diagnoses entirely
            if is_historical(diagnosis) or is_chronic(diagnosis):
                continue
            key = normalize_diagnosis(diagnosis)
            if key not in seen_diagnoses:
                seen_diagnoses[key] = {"hadm_id": note["hadm_id"], "raw": diagnosis}

    seen = set()
    drift = []
    for note in sorted_notes:
        text_lower = note["text"].lower()
        for key, source in seen_diagnoses.items():
            if note["hadm_id"] == source["hadm_id"]:
                continue
            for keyword in resolved_keywords:
                if keyword in text_lower and key in text_lower:
                    dedup_key = (key, note["hadm_id"])
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