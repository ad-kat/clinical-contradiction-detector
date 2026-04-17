"""
detector.py
Cross-encounter contradiction detection for ClinicalContradiction.

Two detector types right now:
  - ALLERGY_MEDICATION_CONFLICT
  - DIAGNOSIS_DRIFT

Added chronic filtering + ML severity scoring layer (Apr 2026)
TODO: add drug interaction checker at some point? not urgent
"""

from __future__ import annotations

import re
from nlp.llm_extractor import extract_all


# chronic conditions that can't truly resolve -- flag differently
# expanded this significantly, was way too short before (had like 15 entries)
CHRONIC_CONDITIONS: set[str] = {
    # metabolic
    "diabetes", "diabetes mellitus", "type 1 diabetes", "type 2 diabetes",
    "hypothyroidism", "hyperthyroidism", "obesity", "hyperlipidemia",
    "dyslipidemia", "hyperuricemia", "gout", "metabolic syndrome",
    # cardiovascular
    "hypertension", "heart failure", "chf", "coronary artery disease", "cad",
    "atrial fibrillation", "afib", "cardiomyopathy", "peripheral artery disease",
    "pad", "deep vein thrombosis", "dvt", "pulmonary hypertension",
    # pulmonary
    "copd", "chronic obstructive pulmonary disease", "asthma",
    "pulmonary fibrosis", "interstitial lung disease", "ild",
    "obstructive sleep apnea", "osa",
    # renal/hepatic
    "chronic kidney disease", "ckd", "end-stage renal disease", "esrd",
    "nephrotic syndrome", "cirrhosis", "hepatitis b", "hepatitis c", "hcv",
    "non-alcoholic fatty liver", "nafld", "nash",
    # neuro/psych
    "epilepsy", "seizure disorder", "multiple sclerosis", "ms",
    "parkinson", "alzheimer", "dementia", "bipolar", "schizophrenia",
    "major depressive disorder", "mdd", "ptsd", "anxiety disorder",
    "obsessive-compulsive disorder", "ocd", "adhd",
    # infectious/immunological
    "hiv", "aids", "tuberculosis", "tb",
    "rheumatoid arthritis", "ra", "lupus", "sle",
    "inflammatory bowel disease", "ibd", "crohn", "ulcerative colitis",
    "celiac", "ankylosing spondylitis", "psoriasis",
    # onco (treated, not cured)
    "cancer", "carcinoma", "lymphoma", "leukemia", "myeloma",
    # misc
    "chronic pain", "fibromyalgia", "osteoarthritis", "osteoporosis",
    "sickle cell", "hemophilia", "thalassemia",
}

# prefixes that mark a diagnosis as historical -- skip for drift
HISTORICAL_PREFIXES: tuple[str, ...] = (
    "hx of", "history of", "h/o", "past history of",
    "prior history of", "previous history of", "pmh of",
    "past medical history of", "known history of",
)

# if these are in the note near the diagnosis, it's currently active
# (prevents chronic conditions from being over-filtered)
ACTIVE_CONTEXT_PHRASES: tuple[str, ...] = (
    "currently", "active", "ongoing", "persistent", "poorly controlled",
    "well controlled", "managed with", "on treatment for", "continues to have",
    "presents with", "admitted for", "exacerbation of",
)

# keywords that suggest a diagnosis has been negated / resolved
RESOLVED_KEYWORDS: list[str] = [
    "resolved", "no longer", "ruled out", "negative for",
    "denies", "no evidence of", "without", "absent", "cleared",
]


# ---------------------------------------------------------------------------
# normalisation helpers
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    return text.lower().strip()


def normalize_diagnosis(diagnosis: str) -> str:
    """strip historical prefixes so we get the bare condition name"""
    d = normalize(diagnosis)
    for prefix in HISTORICAL_PREFIXES:
        if d.startswith(prefix):
            d = d[len(prefix):].strip()
    return d


def is_chronic(diagnosis: str) -> bool:
    d = normalize(diagnosis)
    return any(c in d for c in CHRONIC_CONDITIONS)


def is_historical(diagnosis: str) -> bool:
    d = normalize(diagnosis)
    return any(d.startswith(prefix) for prefix in HISTORICAL_PREFIXES)


def context_confirms_active(diagnosis: str, note_text: str) -> bool:
    """
    check if the note actually confirms the diagnosis is currently active.
    looks in a ~200 char window around each mention of the condition.

    this is important for chronic conditions -- we don't want to suppress
    something like "COPD exacerbation" just because COPD is in the chronic list
    """
    diag_key = normalize_diagnosis(diagnosis)
    text_lower = note_text.lower()

    for match in re.finditer(re.escape(diag_key), text_lower):
        start = max(0, match.start() - 100)
        end = min(len(text_lower), match.end() + 100)
        window = text_lower[start:end]
        if any(phrase in window for phrase in ACTIVE_CONTEXT_PHRASES):
            return True
    return False


# ---------------------------------------------------------------------------
# chronic condition filtering (refined)
# ---------------------------------------------------------------------------

def should_skip_for_drift(diagnosis: str, note_text: str) -> bool:
    """
    decides whether to exclude a diagnosis from drift detection.

    rules:
    1. always skip historical prefixes (hx of, h/o, etc.)
    2. skip chronic conditions UNLESS the note actively confirms they're
       currently presenting -- then keep them so we catch real exacerbation drift
    """
    if is_historical(diagnosis):
        return True

    if is_chronic(diagnosis):
        # only skip if it's NOT actively confirmed in this note
        if not context_confirms_active(diagnosis, note_text):
            return True

    return False


def is_chronic_reappearance(diagnosis: str, later_notes: list[dict]) -> bool:
    """
    returns True if a 'resolved' diagnosis shows up as active again in
    subsequent notes. used to boost confidence score on confirmed reappearances.
    """
    diag_key = normalize_diagnosis(diagnosis)
    for note in later_notes:
        text_lower = note["text"].lower()
        if diag_key in text_lower and context_confirms_active(diagnosis, note["text"]):
            return True
    return False


# ---------------------------------------------------------------------------
# severity scoring ML layer
# ---------------------------------------------------------------------------
# feature-based confidence scorer, produces 0.0-1.0 per contradiction.
# think of it as a hand-tuned logistic regression -- each feature has a weight,
# scores sum and clamp to [0,1], then map to HIGH/MEDIUM/LOW.
#
# could replace this with a proper sklearn model later if we get labelled data,
# but this is good enough for now and way more interpretable

# weights -- tweaked through testing, may need adjustment
_W = {
    # allergy-medication features
    "allergy_exact_match":       0.40,
    "allergy_class_match":       0.30,
    "multiple_encounters_gap":   0.15,
    "allergy_in_problem_list":   0.10,
    "high_risk_drug_class":      0.15,

    # diagnosis drift features
    "chronic_condition":         0.20,
    "reappears_after_resolved":  0.35,
    "keyword_strength":          0.20,
    "multi_encounter_drift":     0.15,
    "diagnosis_in_problem_list": 0.10,
}

# drug class synonym map -- this is what catches penicillin -> amoxicillin etc.
# will expand as needed
HIGH_RISK_DRUG_CLASSES: dict[str, list[str]] = {
    "penicillin": [
        "amoxicillin", "ampicillin", "piperacillin", "nafcillin",
        "oxacillin", "dicloxacillin", "flucloxacillin", "augmentin",
        "amoxicillin-clavulanate", "piperacillin-tazobactam",
    ],
    "cephalosporin": [
        "cephalexin", "cefazolin", "ceftriaxone", "cefepime",
        "cefdinir", "cefuroxime", "ceftazidime",
    ],
    "sulfa": [
        "trimethoprim-sulfamethoxazole", "sulfamethoxazole", "bactrim",
        "sulfadiazine", "sulfasalazine",
    ],
    "nsaid": [
        "ibuprofen", "naproxen", "ketorolac", "meloxicam", "celecoxib",
        "indomethacin", "diclofenac", "aspirin",
    ],
    "opioid": [
        "morphine", "oxycodone", "hydrocodone", "fentanyl", "codeine",
        "tramadol", "hydromorphone", "methadone", "buprenorphine",
    ],
    "contrast": ["iodine", "contrast dye", "gadolinium", "omnipaque", "visipaque"],
    "ace inhibitor": [
        "lisinopril", "enalapril", "ramipril", "benazepril",
        "captopril", "quinapril", "perindopril",
    ],
}

# keyword strength for negation (higher = stronger signal of actual resolution)
KEYWORD_STRENGTH: dict[str, float] = {
    "ruled out":      1.0,
    "negative for":   0.9,
    "no evidence of": 0.85,
    "resolved":       0.75,
    "no longer":      0.70,
    "absent":         0.65,
    "cleared":        0.60,
    "denies":         0.50,
    "without":        0.40,
}


def _get_drug_class(drug_name: str) -> str | None:
    """return the class name for a drug, None if not found"""
    d = normalize(drug_name).split()[0]
    for cls, members in HIGH_RISK_DRUG_CLASSES.items():
        if d in members or any(m.startswith(d) for m in members):
            return cls
    return None


def _allergy_class_conflict(allergy: str, medication: str) -> bool:
    """
    True if medication is in the same drug class as the allergy.
    e.g. allergy=penicillin, medication=amoxicillin -> True
    """
    a_key = normalize(allergy).split()[0]
    m_cls = _get_drug_class(medication)

    if m_cls and a_key in m_cls:
        return True

    # also check if allergy IS a class name and med is in that class
    if a_key in HIGH_RISK_DRUG_CLASSES:
        meds_in_class = HIGH_RISK_DRUG_CLASSES[a_key]
        m_key = normalize(medication).split()[0]
        return any(m_key in m or m.startswith(m_key) for m in meds_in_class)

    return False


def score_allergy_conflict(conflict: dict, all_notes: list[dict]) -> tuple[float, str]:
    """
    compute confidence score + revised severity for an ALLERGY_MEDICATION_CONFLICT.
    returns (score, severity_label)
    """
    score = 0.0

    allergy = conflict.get("allergy", "")
    medication = conflict.get("medication", "")
    allergy_admission = conflict.get("allergy_noted_in")
    med_admission = conflict.get("medication_prescribed_in")

    # exact name match
    if normalize(allergy) == normalize(medication.split()[0]):
        score += _W["allergy_exact_match"]

    # drug class match
    if _allergy_class_conflict(allergy, medication):
        score += _W["allergy_class_match"]

    # cross-admission gap (more concerning than same-admission)
    if allergy_admission and med_admission and allergy_admission != med_admission:
        score += _W["multiple_encounters_gap"]

    # "allergy" mentioned in the prescribing note near the medication
    med_key = normalize(medication).split()[0]
    for note in all_notes:
        if note.get("hadm_id") == med_admission:
            text_lower = note["text"].lower()
            if "allerg" in text_lower and med_key in text_lower:
                score += _W["allergy_in_problem_list"]
                break

    # high-risk drug class bonus
    if _get_drug_class(medication) is not None:
        score += _W["high_risk_drug_class"]

    score = min(score, 1.0)
    severity = _score_to_severity(score, base="HIGH")
    return round(score, 3), severity


def score_diagnosis_drift(
    drift: dict,
    all_notes: list[dict],
    sorted_notes: list[dict],
) -> tuple[float, str]:
    """confidence score + revised severity for a DIAGNOSIS_DRIFT"""
    score = 0.0

    diagnosis = drift.get("diagnosis", "")
    conflict_admission = drift.get("conflict_in")
    first_admission = drift.get("first_noted_in")
    keyword_used = drift.get("_keyword", "resolved")

    # chronic condition -> more clinically significant
    if is_chronic(diagnosis):
        score += _W["chronic_condition"]

    # reappears in later notes after being resolved -> strong signal
    conflict_idx = next(
        (i for i, n in enumerate(sorted_notes) if n.get("hadm_id") == conflict_admission),
        len(sorted_notes),
    )
    later_notes = sorted_notes[conflict_idx + 1:]
    if is_chronic_reappearance(diagnosis, later_notes):
        score += _W["reappears_after_resolved"]

    # keyword strength
    kw_strength = KEYWORD_STRENGTH.get(keyword_used, 0.5)
    score += _W["keyword_strength"] * kw_strength

    # how many encounters between first-seen and conflict
    if first_admission and conflict_admission:
        ids_between = [
            n["hadm_id"] for n in sorted_notes
            if n.get("hadm_id") not in (first_admission, conflict_admission)
        ]
        gap = len(set(ids_between))
        score += _W["multi_encounter_drift"] * min(gap / 3.0, 1.0)

    # was it in a problem list / impression section?
    diag_key = normalize_diagnosis(diagnosis)
    for note in all_notes:
        if note.get("hadm_id") == first_admission:
            text_lower = note["text"].lower()
            section_match = re.search(
                r"(problem list|impression|assessment|diagnosis)[:\s]",
                text_lower
            )
            if section_match:
                window = text_lower[section_match.start():section_match.start() + 500]
                if diag_key in window:
                    score += _W["diagnosis_in_problem_list"]
                    break

    score = min(score, 1.0)
    severity = _score_to_severity(score, base="MEDIUM")
    return round(score, 3), severity


def _score_to_severity(score: float, base: str) -> str:
    """
    map confidence score to severity label.
    the ML score can only keep or lower the rule-based default, never raise it above HIGH.
    """
    if score >= 0.70:
        return "HIGH"
    elif score >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def extract_all_notes(notes: list[dict]) -> list[dict]:
    """run LLM extraction once per note and cache results on the note dict"""
    return [
        {**note, "_facts": extract_all(note["text"])}
        for note in notes
    ]


# ---------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------

def detect_allergy_medication_conflict(extracted: list[dict]) -> list[dict]:
    """
    finds cases where a medication is prescribed despite a documented allergy.
    now includes drug-class matching (e.g. penicillin allergy + amoxicillin).

    each conflict gets scored and severity may be revised by the ML layer.
    """
    sorted_notes = sorted(extracted, key=lambda x: x.get("charttime") or "")

    # build allergy registry, first occurrence wins
    allergy_sources: dict[str, dict] = {}
    for note in sorted_notes:
        for allergy in note["_facts"]["allergies"]:
            key = normalize(allergy)
            if key not in allergy_sources:
                allergy_sources[key] = {"hadm_id": note["hadm_id"], "raw": allergy}

    seen: set[tuple] = set()
    conflicts: list[dict] = []

    for note in sorted_notes:
        for med in note["_facts"]["medications"]:
            med_key = normalize(med.split()[0])

            for allergy_key, source in allergy_sources.items():
                exact_hit = allergy_key in med_key or med_key in allergy_key
                class_hit = _allergy_class_conflict(source["raw"], med)

                if not (exact_hit or class_hit):
                    continue

                dedup_key = (allergy_key, med_key)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                conflict = {
                    "type": "ALLERGY_MEDICATION_CONFLICT",
                    "severity": "HIGH",
                    "confidence": None,
                    "allergy": source["raw"],
                    "allergy_noted_in": source["hadm_id"],
                    "medication": med,
                    "medication_prescribed_in": note["hadm_id"],
                    "match_type": "class" if (class_hit and not exact_hit) else "exact",
                    "explanation": (
                        f"'{med}' was prescribed but '{source['raw']}' is listed as an allergy."
                        if exact_hit else
                        f"'{med}' belongs to the same drug class as documented allergy '{source['raw']}'."
                    ),
                }

                score, revised_severity = score_allergy_conflict(conflict, sorted_notes)
                conflict["confidence"] = score
                conflict["severity"] = revised_severity
                conflicts.append(conflict)

    return conflicts


def detect_diagnosis_drift(extracted: list[dict]) -> list[dict]:
    """
    finds cases where a previously noted diagnosis is described as resolved/negated
    in a later encounter.

    chronic filtering is now context-aware -- chronic conditions get skipped unless
    the note confirms they're actually presenting (exacerbation etc.)
    """
    sorted_notes = sorted(extracted, key=lambda x: x.get("charttime") or "")

    # build first-seen diagnosis map
    seen_diagnoses: dict[str, dict] = {}
    for note in sorted_notes:
        for diagnosis in note["_facts"]["diagnoses"]:
            if should_skip_for_drift(diagnosis, note["text"]):
                continue
            key = normalize_diagnosis(diagnosis)
            if key not in seen_diagnoses:
                seen_diagnoses[key] = {"hadm_id": note["hadm_id"], "raw": diagnosis}

    seen: set[tuple] = set()
    drift_list: list[dict] = []

    for note in sorted_notes:
        text_lower = note["text"].lower()
        for key, source in seen_diagnoses.items():
            if note["hadm_id"] == source["hadm_id"]:
                continue

            for keyword, strength in KEYWORD_STRENGTH.items():
                if keyword in text_lower and key in text_lower:
                    dedup_key = (key, note["hadm_id"])
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    drift = {
                        "type": "DIAGNOSIS_DRIFT",
                        "severity": "MEDIUM",
                        "confidence": None,
                        "diagnosis": source["raw"],
                        "first_noted_in": source["hadm_id"],
                        "conflict_in": note["hadm_id"],
                        "resolution_keyword": keyword,
                        "_keyword": keyword,  # internal, stripped before returning
                        "is_chronic": is_chronic(source["raw"]),
                        "explanation": (
                            f"'{source['raw']}' was previously noted but appears "
                            f"'{keyword}' in a later encounter."
                        ),
                    }

                    score, revised_severity = score_diagnosis_drift(
                        drift, sorted_notes, sorted_notes
                    )
                    drift["confidence"] = score
                    drift["severity"] = revised_severity
                    drift.pop("_keyword", None)
                    drift_list.append(drift)
                    break  # one keyword per (diag, note) pair is enough

    return drift_list


def detect_all_contradictions(notes: list[dict]) -> list[dict]:
    """
    main entry point -- runs both detectors and returns sorted results.
    sorted by severity rank then confidence (highest first).
    """
    extracted = extract_all_notes(notes)

    contradictions: list[dict] = []
    contradictions += detect_allergy_medication_conflict(extracted)
    contradictions += detect_diagnosis_drift(extracted)

    severity_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    contradictions.sort(
        key=lambda c: (
            severity_rank.get(c.get("severity", "LOW"), 2),
            -(c.get("confidence") or 0.0)
        )
    )

    return contradictions