# app/extractor/schemas.py
"""
Canonical type contracts for the AI-Adaptive Onboarding Engine extractor pipeline.
All inter-module data exchange must conform to these TypedDicts.
"""

from typing import TypedDict, List, Optional


class SkillEntry(TypedDict):
    """
    Final output contract for a single mapped and scored skill.
    This is the shape consumed by downstream API routes and the database layer.

    Fields:
        esco_uri      : Canonical ESCO skill URI or internal ID.
        label         : Human-readable ESCO skill label.
        mastery_score : Bloom's Mastery Learning score, strictly 0.0 – 1.0.
    """
    esco_uri     : str
    label        : str
    mastery_score: float


class ESCOEntry(TypedDict):
    """
    Shape of a single entry in the ESCO catalog dict passed to skillsim_model.
    Keys are ESCO URIs, values are ESCOEntry dicts.

    Example catalog:
        {
            "http://data.europa.eu/esco/skill/abc123": {
                "label": "Python (programming language)",
                "description": "..."   # optional, unused by sim model
            }
        }
    """
    label      : str
    description: Optional[str]