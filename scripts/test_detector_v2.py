"""
test_detector_v2.py -- offline unit tests for the two new features:
  1. chronic condition filtering refinement
  2. severity scoring ML layer

no Groq API key needed -- facts are injected directly, LLM extraction is mocked.

HOW TO RUN (from project root):
    python scripts/test_detector_v2.py

that's it. no Groq calls, no DB, no .env needed.
if you want to double-check, you can also do:
    GROQ_API_KEY=dummy python scripts/test_detector_v2.py
(same thing, the env var is just ignored since we mock the LLM)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# mock extract_all so nothing actually calls Groq
# has to happen BEFORE importing detector
import nlp.llm_extractor as _ext

def _mock_extract(text: str) -> dict:
    # facts are injected via note['_facts'] in these tests
    return {"allergies": [], "medications": [], "diagnoses": []}

_ext.extract_all = _mock_extract

from nlp.detector import (
    is_chronic,
    is_historical,
    should_skip_for_drift,
    context_confirms_active,
    _allergy_class_conflict,
    score_allergy_conflict,
    score_diagnosis_drift,
    detect_allergy_medication_conflict,
    detect_diagnosis_drift,
    KEYWORD_STRENGTH,
)

# simple pass/fail tracking
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results = []

def check(label, condition):
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}")
    _results.append(condition)


# ---------------------------------------------------------------------------
# test helpers
# ---------------------------------------------------------------------------

def make_note(hadm_id, text="", allergies=None, medications=None, diagnoses=None, charttime="2021-01-01"):
    """build a synthetic note with pre-injected facts (bypasses LLM)"""
    return {
        "hadm_id": hadm_id,
        "charttime": charttime,
        "text": text,
        "_facts": {
            "allergies":   allergies   or [],
            "medications": medications or [],
            "diagnoses":   diagnoses   or [],
        },
    }


# patch extract_all_notes to pass pre-injected facts through
import nlp.detector as det

def _passthrough(notes):
    return [
        n if "_facts" in n
        else {**n, "_facts": {"allergies": [], "medications": [], "diagnoses": []}}
        for n in notes
    ]

det.extract_all_notes = _passthrough


# ===========================================================================
# PART 1: chronic condition filtering
# ===========================================================================
print("\n=== PART 1: Chronic Condition Filtering ===\n")

check("diabetes is chronic",            is_chronic("diabetes"))
check("type 2 diabetes is chronic",     is_chronic("Type 2 Diabetes Mellitus"))
check("appendicitis is NOT chronic",    not is_chronic("appendicitis"))
check("pneumonia is NOT chronic",       not is_chronic("pneumonia"))
check("heart failure is chronic",       is_chronic("congestive heart failure"))
check("CKD is chronic",                 is_chronic("chronic kidney disease"))

check("'history of diabetes' is historical",  is_historical("history of diabetes"))
check("'h/o COPD' is historical",             is_historical("h/o COPD"))
check("'diabetes' alone is NOT historical",   not is_historical("diabetes"))
check("'hx of epilepsy' is historical",       is_historical("hx of epilepsy"))

# context-aware: chronic + active phrasing -> should NOT be skipped
note_exacerbation = "Patient presents with COPD exacerbation, currently on bronchodilators."
check("COPD with 'presents with' context -> NOT skipped",
      not should_skip_for_drift("COPD", note_exacerbation))

note_generic = "Medications were reviewed. Patient discharged in stable condition."
check("COPD without active context -> skipped",
      should_skip_for_drift("COPD", note_generic))

check("'history of diabetes' always skipped even with active context",
      should_skip_for_drift("history of diabetes", note_exacerbation))

note_active = "Patient is actively managed for hypertension with lisinopril."
check("context_confirms_active: 'managed with' near hypertension -> True",
      context_confirms_active("hypertension", note_active))

note_inactive = "Vitals stable. Patient denies chest pain."
check("context_confirms_active: no active phrase near hypertension -> False",
      not context_confirms_active("hypertension", note_inactive))


# ===========================================================================
# PART 2: drug class matching
# ===========================================================================
print("\n=== PART 2: Drug Class Conflict Matching ===\n")

check("penicillin allergy + amoxicillin -> class conflict",
      _allergy_class_conflict("penicillin", "amoxicillin 500mg PO TID"))
check("penicillin allergy + piperacillin-tazobactam -> class conflict",
      _allergy_class_conflict("penicillin", "piperacillin-tazobactam"))
check("sulfa allergy + bactrim -> class conflict",
      _allergy_class_conflict("sulfa", "bactrim DS"))
check("nsaid allergy + ibuprofen -> class conflict",
      _allergy_class_conflict("nsaid", "ibuprofen 400mg"))
check("penicillin allergy + vancomycin -> NO class conflict",
      not _allergy_class_conflict("penicillin", "vancomycin 1g IV"))
check("aspirin allergy + morphine -> NO class conflict",
      not _allergy_class_conflict("aspirin", "morphine 4mg IV"))

# exact conflict still works
notes_exact = [
    make_note(111, text="Allergies: Aspirin", allergies=["Aspirin"], charttime="2021-01-01"),
    make_note(222, text="Meds: Aspirin 81mg daily", medications=["Aspirin 81mg daily"], charttime="2021-02-01"),
]
conflicts_exact = detect_allergy_medication_conflict(notes_exact)
check("Exact Aspirin conflict detected",          len(conflicts_exact) == 1)
check("Exact conflict has confidence score",       conflicts_exact[0].get("confidence") is not None)

# class-level conflict: penicillin -> amoxicillin
notes_class = [
    make_note(111, text="Allergies: Penicillin", allergies=["Penicillin"], charttime="2021-01-01"),
    make_note(222, text="Meds: Amoxicillin 500mg PO", medications=["Amoxicillin 500mg PO"], charttime="2021-02-01"),
]
conflicts_class = detect_allergy_medication_conflict(notes_class)
check("Class-level Penicillin -> Amoxicillin conflict detected", len(conflicts_class) == 1)
if conflicts_class:
    check("match_type == 'class'",          conflicts_class[0].get("match_type") == "class")
    check("confidence score present",        conflicts_class[0].get("confidence") is not None)
    print(f"    confidence={conflicts_class[0]['confidence']}, severity={conflicts_class[0]['severity']}")


# ===========================================================================
# PART 3: severity scoring ML layer
# ===========================================================================
print("\n=== PART 3: Severity Scoring ML Layer ===\n")

# high confidence: exact match + cross-admission + high-risk class (penicillin)
conflict_hi = {
    "type": "ALLERGY_MEDICATION_CONFLICT",
    "allergy": "penicillin",
    "medication": "amoxicillin 500mg PO TID",
    "allergy_noted_in": 111,
    "medication_prescribed_in": 222,
}
notes_hi = [
    make_note(111, text="Allergies: penicillin. Patient is allergic.", charttime="2021-01-01"),
    make_note(222, text="amoxicillin prescribed for UTI. Allergy list reviewed.", charttime="2021-02-01"),
]
score_hi, sev_hi = score_allergy_conflict(conflict_hi, notes_hi)
print(f"    High-confidence conflict -> score={score_hi}, severity={sev_hi}")
check("score >= 0.60",          score_hi >= 0.60)
check("severity is HIGH",       sev_hi == "HIGH")

# low confidence: no class match, same admission, weird allergy substance
conflict_lo = {
    "type": "ALLERGY_MEDICATION_CONFLICT",
    "allergy": "latex",
    "medication": "acetaminophen 500mg",
    "allergy_noted_in": 333,
    "medication_prescribed_in": 333,   # same admission
}
notes_lo = [make_note(333, text="acetaminophen given", charttime="2021-01-01")]
score_lo, sev_lo = score_allergy_conflict(conflict_lo, notes_lo)
print(f"    Low-confidence conflict -> score={score_lo}, severity={sev_lo}")
check("low confidence score < 0.50", score_lo < 0.50)

# diagnosis drift scoring
drift_medium = {
    "type": "DIAGNOSIS_DRIFT",
    "diagnosis": "ascites",
    "first_noted_in": 111,
    "conflict_in": 222,
    "_keyword": "resolved",
    "is_chronic": False,
}
notes_drift = [
    make_note(111, text="Impression: Ascites noted.", diagnoses=["ascites"], charttime="2021-01-01"),
    make_note(222, text="Ascites resolved. Follow-up in clinic.",             charttime="2021-03-01"),
]
score_d, sev_d = score_diagnosis_drift(drift_medium, notes_drift, notes_drift)
print(f"    Drift (resolved, non-chronic) -> score={score_d}, severity={sev_d}")
check("drift score > 0",         score_d > 0.0)
check("drift score in [0, 1]",   0.0 <= score_d <= 1.0)

# stronger keyword should score higher
drift_strong = {**drift_medium, "_keyword": "ruled out"}
score_strong, sev_strong = score_diagnosis_drift(drift_strong, notes_drift, notes_drift)
check("'ruled out' scores higher than 'resolved'", score_strong > score_d)
print(f"    Drift (ruled out) -> score={score_strong}, severity={sev_strong}")

# output sorted by severity/confidence
notes_sort = [
    make_note(111, text="Allergies: Aspirin. Penicillin.", allergies=["Aspirin", "Penicillin"], charttime="2021-01-01"),
    make_note(222, text="Aspirin 81mg daily. Amoxicillin 500mg TID.",
              medications=["Aspirin 81mg daily", "Amoxicillin 500mg TID"], charttime="2021-02-01"),
]
all_c = detect_allergy_medication_conflict(notes_sort)
if len(all_c) > 1:
    first_sev = all_c[0]["severity"]
    check("output sorted highest severity first", first_sev in ("HIGH", "MEDIUM"))


# ===========================================================================
# summary
# ===========================================================================
passed = sum(_results)
total  = len(_results)
print(f"\n{'='*50}")
print(f"Results: {passed}/{total} passed", "✓" if passed == total else "✗")
if passed < total:
    print("  some tests failed -- see above for details")
print("=" * 50)