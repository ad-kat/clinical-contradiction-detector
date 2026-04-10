"""
LLM-based clinical fact extractor using Groq + llama-3.3-70b-versatile.
Drop-in replacement for nlp/extractor.py — same extract_all() interface.
"""

import json
import re
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a clinical NLP system. Extract medical facts from clinical notes.
Return ONLY valid JSON with exactly these keys: allergies, medications, diagnoses.
No markdown, no explanation, no preamble.

Schema:
{
  "allergies": ["substance name", ...],
  "medications": ["Drug dose frequency", ...],
  "diagnoses": ["condition name", ...]
}

Rules:
- allergies: substances the patient is allergic to. Empty list if none or NKDA.
- medications: include admission meds, discharge meds, current meds. Format: "DrugName dose frequency" where available.
- diagnoses: active or historical conditions, findings, impressions. Include radiology impressions.
- If a field has nothing, return empty list [].
- Never return null. Always return all three keys.
"""

def _call_llm(note_text: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract facts from this note:\n\n{note_text.strip()}"}
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def extract_all(text: str) -> dict:
    """
    Same interface as the rule-based extractor.
    Returns: {"allergies": [...], "medications": [...], "diagnoses": [...]}
    """
    try:
        result = _call_llm(text)
        # Guarantee all keys exist
        return {
            "allergies": result.get("allergies", []),
            "medications": result.get("medications", []),
            "diagnoses": result.get("diagnoses", []),
        }
    except (json.JSONDecodeError, Exception):
        # Fallback to empty rather than crash
        return {"allergies": [], "medications": [], "diagnoses": []}


# Keep individual functions for backward compat
def extract_allergies(text: str) -> list:
    return extract_all(text)["allergies"]

def extract_medications(text: str, section: str = "both") -> list:
    return extract_all(text)["medications"]

def extract_diagnoses(text: str) -> list:
    return extract_all(text)["diagnoses"]