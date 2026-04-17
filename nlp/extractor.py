import re
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_allergies(text: str) -> list[str]:
    allergies = []
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
            block = match.group(1)
            meds += parse_med_block(block)
    if section in ("discharge", "both"):
        pattern = r'DISCHARGE MEDICATIONS?:(.+?)(?:\n[A-Z]|\Z)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            block = match.group(1)
            meds += parse_med_block(block)
    return list(set(meds))

def parse_med_block(block: str) -> list[str]:
    meds = []
    for line in block.strip().split("\n"):
        line = line.strip().lstrip("0123456789.-) ")
        if len(line) > 3:
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
    return {
        "allergies": extract_allergies(text),
        "medications": extract_medications(text),
        "diagnoses": extract_diagnoses(text),
    }
