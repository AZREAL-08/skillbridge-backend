"""
Set intersection logic for skill gap analysis.
Skill Gap URIs = Target JD URIs − User Mastered URIs
"""

from typing import List
from app.state import SkillEntry

# Mastery threshold: >= 0.85 means "mastered" (bypassed)
MASTERY_THRESHOLD = 0.85


def compute_skill_gap(
    extracted_skills: List[SkillEntry],
    required_uris: List[str],
) -> List[str]:
    """
    Return the list of ESCO URIs the user still needs to learn.
    A skill is considered mastered if mastery_score >= MASTERY_THRESHOLD.
    """
    mastered_uris = {
        s["esco_uri"]
        for s in extracted_skills
        if s["mastery_score"] >= MASTERY_THRESHOLD
    }
    return [uri for uri in required_uris if uri not in mastered_uris]
