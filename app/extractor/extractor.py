"""
Public interface: extract_skills()
Cascading NLP pipeline:
  1. SkillNer  — explicit extraction (rule-based, spaCy)
  2. JobBERT   — implicit extraction (token classification)
  3. Skill-Sim — taxonomic mapping (sentence-transformer → ESCO URIs)
  4. Groq      — mastery inference (Llama-3-8B, JSON-mode)

Multi-Model Confidence Scoring:
  confidence = 0.5 * skillner_score + 0.3 * jobbert_score + 0.2 * similarity_score
"""

from typing import List
from app.state import SkillEntry


def compute_confidence(
    skillner_score: float,
    jobbert_score: float,
    similarity_score: float,
) -> float:
    """Weighted multi-model confidence."""
    return 0.5 * skillner_score + 0.3 * jobbert_score + 0.2 * similarity_score


def extract_skills(resume_text: str) -> List[SkillEntry]:
    """
    Run the full extraction pipeline on raw resume text.
    Returns a list of SkillEntry dicts with ESCO URIs, labels,
    mastery scores, and confidence scores.
    """
    # TODO: wire up individual model modules
    raise NotImplementedError("Extraction pipeline not yet implemented")
