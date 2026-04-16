import time
from groq import Groq, RateLimitError
from dotenv import load_dotenv
import json, re, os

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

def _call_llm(note_text: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
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
        except RateLimitError:
            wait = 10 * (attempt + 1)
            print(f"[rate limit] waiting {wait}s before retry {attempt+1}/{retries}")
            time.sleep(wait)
        except Exception as e:
            print(f"[llm error] {e}")
            break
    return {"allergies": [], "medications": [], "diagnoses": []}

def extract_all(text: str) -> dict:
    try:
        result = _call_llm(text)
        return {
            "allergies": result.get("allergies", []),
            "medications": result.get("medications", []),
            "diagnoses": result.get("diagnoses", []),
        }
    except Exception:
        return {"allergies": [], "medications": [], "diagnoses": []}

def extract_allergies(text: str) -> list:
    return extract_all(text)["allergies"]

def extract_medications(text: str, section: str = "both") -> list:
    return extract_all(text)["medications"]

def extract_diagnoses(text: str) -> list:
    return extract_all(text)["diagnoses"]