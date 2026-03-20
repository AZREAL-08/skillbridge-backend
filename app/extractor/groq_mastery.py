import json
import logging
import os
from typing import List
from groq import Groq

logger = logging.getLogger(__name__)

_GROQ_MODEL    = "llama-3.3-70b-versatile"
_GROQ_CLIENT   = None

def _get_groq_client() -> Groq:
    global _GROQ_CLIENT
    if _GROQ_CLIENT is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("[GroqMastery] GROQ_API_KEY environment variable is not set.")
        _GROQ_CLIENT = Groq(api_key=api_key)
    return _GROQ_CLIENT

_SYSTEM_PROMPT = """
You are a strict talent assessment engine. Your sole task is to analyze a candidate's resume text and assign a mastery_score for each provided skill based on Benjamin Bloom's framework:
  0.0 – 0.2  : Awareness     — mentioned in passing, no demonstrated use
  0.2 – 0.4  : Foundational  — bootcamp, tutorial, or single toy project
  0.4 – 0.6  : Competent     — used in at least one real professional project
  0.6 – 0.8  : Proficient    — multiple years of professional use, clear ownership
  0.8 – 1.0  : Expert        — led teams, architected systems, or mentored others

STRICT RULES:
  1. Base score ONLY on evidence in the resume_text. If no context, assign 0.2.
  2. mastery_score MUST be a float between 0.0 and 1.0.
  3. You MUST respond with a JSON object containing a single key "skills" which holds an array of the results.

OUTPUT FORMAT:
{
  "skills": [
    {"taxonomy_id": "<id>", "label": "<label>", "mastery_score": <float>}
  ]
}
""".strip()

def estimate_mastery(resume_text: str, mapped_skills: List[dict]) -> List[dict]:
    if not mapped_skills:
        return []
    if not resume_text or not resume_text.strip():
        return _apply_fallback(mapped_skills)

    skill_payload = [{"taxonomy_id": s["taxonomy_id"], "label": s["label"]} for s in mapped_skills]
    user_message = f"RESUME TEXT:\n{resume_text.strip()}\n\nSKILLS TO SCORE:\n{json.dumps(skill_payload, ensure_ascii=False, indent=2)}"

    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=_GROQ_MODEL,
            temperature=0.0,
            max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )
        raw_content = response.choices[0].message.content.strip()
        return _parse_and_validate(raw_content, mapped_skills)
    except Exception as exc:
        logger.error("[GroqMastery] LLM call failed: %s", exc)
        return _apply_fallback(mapped_skills)

def _parse_and_validate(raw_content: str, mapped_skills: List[dict]) -> List[dict]:
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        return _apply_fallback(mapped_skills)

    # Extract from the expected "skills" wrapper
    skills_array = parsed.get("skills", [])
    if not isinstance(skills_array, list):
        return _apply_fallback(mapped_skills)

    validated = []
    for entry in skills_array:
        if not isinstance(entry, dict):
            continue
        tax_id = entry.get("taxonomy_id", "")
        label  = entry.get("label", "")
        try:
            score = max(0.0, min(1.0, float(entry.get("mastery_score", 0.3))))
        except (TypeError, ValueError):
            score = 0.3
        
        if tax_id and label:
            # Carry forward taxonomy_source and confidence_score from original mapped_skills
            source = "emsi"
            conf   = 0.5
            for s in mapped_skills:
                if s.get("taxonomy_id") == tax_id:
                    source = s.get("taxonomy_source", "emsi")
                    conf   = s.get("confidence_score", 0.5)
                    break
            validated.append({
                "taxonomy_id": tax_id,
                "taxonomy_source": source,
                "label": label,
                "mastery_score": score,
                "confidence_score": conf,
            })

    if len(validated) < len(mapped_skills):
        validated = _fill_missing(validated, mapped_skills)
    return validated

def _apply_fallback(mapped_skills: List[dict]) -> List[dict]:
    return [
        {
            "taxonomy_id": s.get("taxonomy_id", ""),
            "taxonomy_source": s.get("taxonomy_source", "emsi"),
            "label": s.get("label", ""),
            "mastery_score": 0.3,
            "confidence_score": s.get("confidence_score", 0.5),
        }
        for s in mapped_skills
    ]

def _fill_missing(validated: List[dict], original: List[dict]) -> List[dict]:
    returned_ids = {e["taxonomy_id"] for e in validated}
    for s in original:
        if s.get("taxonomy_id") not in returned_ids:
            validated.append({
                "taxonomy_id": s.get("taxonomy_id", ""),
                "taxonomy_source": s.get("taxonomy_source", "emsi"),
                "label": s.get("label", ""),
                "mastery_score": 0.3,
                "confidence_score": s.get("confidence_score", 0.5),
            })
    return validated