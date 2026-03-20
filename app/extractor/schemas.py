# app/extractor/schemas.py
"""
Canonical type contracts for the AI-Adaptive Onboarding Engine extractor pipeline.
All inter-module data exchange must conform to these TypedDicts.
"""

from typing import TypedDict, Literal


class SkillEntry(TypedDict):
    """
    Final output contract for a single mapped and scored skill.
    This is the shape consumed by downstream API routes and the database layer.

    Fields:
        taxonomy_id      : EMSI skill ID (e.g. "KS440L566SHJ6KQKFHKF") or synthetic
                           "inferred::<span>" for JobBERT-only extractions.
        taxonomy_source  : "emsi" for SkillNER-matched skills, "inferred" for
                           JobBERT spans that couldn't be mapped to an EMSI ID.
        label            : Human-readable skill label.
        mastery_score    : Bloom's Mastery Learning score, strictly 0.0 – 1.0.
        confidence_score : Multi-model weighted confidence, 0.0 – 1.0.
    """
    taxonomy_id      : str
    taxonomy_source  : str      # "emsi" | "inferred"
    label            : str
    mastery_score    : float
    confidence_score : float