"""
extractor.py -- regex-based extraction (scaffold / fallback)

This was the original approach before switching to the LLM extractor.
Keeping it around because:
  1. useful as a fast pre-filter or fallback if Groq is down
  2. some of the section parsing logic might be reusable later

Not actively used in the main pipeline anymore -- llm_extractor.py handles
everything. but don't delete this yet.

# old approach that worked ok for structured notes but fell apart on
# free-text discharge summaries:
#
# def extract_all_v1(text):
#     return {
#         "allergies": extract_allergies(text),
#         "medications": extract_medications(text),
#         "diagnoses": extract_diagnoses(text),
#     }
"""

import re
import spacy

nlp = spacy.load("en_core_web_sm")


def extract_allergies(text: str) -> list[str]:
    allergies = []
    # looks for "ALLERGIES: ..." or "ALLERGY: ..."
    pattern = r'ALLERGIES?:\s*(.+?)(?:\n|$)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        raw = match.group(1).strip()
        if "no known" in raw.lower() or "nkda" in raw.lower():
            return []
        allergies = [a.strip() for a in re.split(r'[,;]', raw) if a.strip()]
    return allergies


def extract_medications(text: str, section: str = "both") -> list[str]:
    meds = []

    if section in ("admission", "both"):
        pattern = r'MEDICATIONS? ON ADMISSION:(.+?)(?:\n[A-Z]|\Z)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            meds += parse_med_block(match.group(1))

    if section in ("discharge", "both"):
        pattern = r'DISCHARGE MEDICATIONS?:(.+?)(?:\n[A-Z]|\Z)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            meds += parse_med_block(match.group(1))

    return list(set(meds))


def parse_med_block(block: str) -> list[str]:
    meds = []
    for line in block.strip().split("\n"):
        line = line.strip().lstrip("0123456789.-) ")
        if len(line) > 3:
            # take everything up to the first number (dose)
            drug = re.split(r'\s+\d', line)[0].strip()
            if drug:
                meds.append(drug)
    return meds


def extract_diagnoses(text: str) -> list[str]:
    diagnoses = []
    pattern = r'DIAGNOSES?:(.+?)(?:\n[A-Z]|\Z)'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        block = match.group(1)
        for line in block.strip().split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if len(line) > 3:
                diagnoses.append(line.strip())
    return diagnoses


def extract_all(text: str) -> dict:
    # NOTE: this is the old regex version -- use llm_extractor.extract_all for real extractions
    return {
        "allergies":   extract_allergies(text),
        "medications": extract_medications(text),
        "diagnoses":   extract_diagnoses(text),
    }