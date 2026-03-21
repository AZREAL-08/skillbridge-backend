import logging
import re
from typing import Dict, List, Set

from app.extractor.schemas       import SkillEntry
from app.extractor.skillner_model import extract_explicit_skills
from app.extractor.jobbert_model  import extract_implicit_skills
from app.extractor.groq_mastery   import compute_mastery_scores

logger = logging.getLogger(__name__)


def extract_skills(raw_text: str, catalog: List[dict] = None) -> List[SkillEntry]:
    """
    Full extraction pipeline for RESUME text.

    Stages:
      1. SkillNER explicit extraction
      2. JobBERT implicit extraction
      3. Merge & deduplicate
      4. Profile-classification mastery scoring (single Groq call + bloom decay)
      5. Cast to SkillEntry

    Args:
        raw_text: Resume text to extract skills from.
        catalog:  Course catalog list (needed for bloom-level mastery decay).
                  If None, mastery defaults to bloom_level=3 for all skills.
    """
    if not raw_text or not raw_text.strip():
        logger.warning("[Orchestrator] Empty input text. Returning [].")
        return []

    logger.info("[Orchestrator] Starting skill extraction pipeline (resume mode).")

    # ── Stage 1: Explicit extraction (SkillNER → EMSI IDs) ───────────────────
    explicit_skills: List[dict] = []
    try:
        explicit_skills = extract_explicit_skills(raw_text)
    except Exception as exc:
        logger.error("[Orchestrator] Stage 1 failed: %s", exc)

    # ── Stage 2: Implicit extraction (JobBERT → EMSI map / inferred) ─────────
    implicit_skills: List[dict] = []
    try:
        implicit_skills = extract_implicit_skills(raw_text)
    except Exception as exc:
        logger.error("[Orchestrator] Stage 2 failed: %s", exc)

    # ── Stage 3: Merge and deduplicate ───────────────────────────────────────
    merged = _merge_and_deduplicate(explicit_skills, implicit_skills)

    if not merged:
        return []

    # ── Stage 4: Profile-classification mastery scoring ───────────────────────
    scored_skills: List[dict] = []
    try:
        scored_skills = compute_mastery_scores(merged, raw_text, catalog or [])
    except Exception as exc:
        logger.error("[Orchestrator] Stage 4 failed: %s", exc)
        # Fallback: assign neutral mastery
        scored_skills = [
            {
                "taxonomy_id": s["taxonomy_id"],
                "taxonomy_source": s["taxonomy_source"],
                "label": s["label"],
                "mastery_score": 0.3,
                "confidence_score": s.get("confidence_score", 0.5),
            }
            for s in merged
        ]

    # ── Stage 5: Cast to strict SkillEntry TypedDict + assign confidence ─────
    return _cast_to_skill_entries(scored_skills)


def _extract_requirements_section(raw_text: str) -> str:
    """
    Extract the "Required Skills" or "Responsibilities" section from a JD.
    
    This avoids extracting skills from company description, benefits text, etc.
    Only returns text after common section headers like:
      - "Required Skills:"
      - "Required Qualifications:"
      - "Key Responsibilities:"
      - "Responsibilities and Qualifications:"
    
    If no such header is found, returns the full text (fallback).
    
    Args:
        raw_text: Full job description text.
        
    Returns:
        Text from the requirements section onward (or full text if no section found).
    """
    if not raw_text:
        return ""
    
    # Common section headers for requirements/skills/responsibilities
    headers = [
        r"(?:^|\n)\s*(?:required\s+)?(?:skills|qualifications|competencies)",
        r"(?:^|\n)\s*(?:key\s+)?(?:responsibilities|responsibilities\s+and\s+qualifications)",
        r"(?:^|\n)\s*what\s+we(?:(?:'|&#39;)re\s+)?(?:looking|seeking)\s+for",
        r"(?:^|\n)\s*(?:essential|must-have|core)\s+(?:skills|qualities)",
        r"(?:^|\n)\s*experience\s+(?:required|needed)",
    ]
    
    text_lower = raw_text.lower()
    earliest_idx = len(raw_text)
    
    # Find the earliest section header
    for header_pattern in headers:
        match = re.search(header_pattern, raw_text, re.IGNORECASE | re.MULTILINE)
        if match:
            earliest_idx = min(earliest_idx, match.start())
    
    # If a header was found, extract everything from that point onward
    if earliest_idx < len(raw_text):
        # Return text starting from the header
        section_text = raw_text[earliest_idx:]
        logger.info("[Orchestrator] Extracted requirements section (%d chars from %d total)",
                   len(section_text), len(raw_text))
        return section_text
    
    # Fallback: use full text if no section header found
    logger.debug("[Orchestrator] No requirements section header found, using full JD text")
    return raw_text


def extract_skills_from_jd(raw_text: str) -> List[SkillEntry]:
    """
    Extraction pipeline for JD (job description) text.

    JDs don't have mastery — they have requirements. This pipeline:
      0. Extract the "Required Skills" section (skip company description)
      1. SkillNER explicit extraction
      2. JobBERT implicit extraction
      3. Merge & deduplicate
      4. SKIP mastery scoring — assign mastery_score = 0.0 (neutral placeholder)
      5. Cast to SkillEntry

    Args:
        raw_text: Job description text to extract required skills from.
    """
    if not raw_text or not raw_text.strip():
        logger.warning("[Orchestrator] Empty JD text. Returning [].")
        return []

    logger.info("[Orchestrator] Starting skill extraction pipeline (JD mode — no mastery).")

    # ── Stage 0: Extract requirements section to avoid company description ───
    jd_requirements = _extract_requirements_section(raw_text)

    # ── Stage 1: Explicit extraction (SkillNER → EMSI IDs) ───────────────────
    explicit_skills: List[dict] = []
    try:
        explicit_skills = extract_explicit_skills(jd_requirements)
    except Exception as exc:
        logger.error("[Orchestrator] Stage 1 (JD) failed: %s", exc)

    # ── Stage 2: Implicit extraction (JobBERT → EMSI map / inferred) ─────────
    implicit_skills: List[dict] = []
    try:
        implicit_skills = extract_implicit_skills(jd_requirements)
    except Exception as exc:
        logger.error("[Orchestrator] Stage 2 (JD) failed: %s", exc)

    # ── Stage 3: Merge and deduplicate ───────────────────────────────────────
    merged = _merge_and_deduplicate(explicit_skills, implicit_skills)

    if not merged:
        return []

    # ── Stage 4: SKIP mastery — JD skills get neutral 0.0 ────────────────────
    jd_skills = [
        {
            "taxonomy_id": s["taxonomy_id"],
            "taxonomy_source": s["taxonomy_source"],
            "label": s["label"],
            "mastery_score": 0.0,
            "confidence_score": s.get("confidence_score", 0.5),
        }
        for s in merged
    ]

    # ── Stage 5: Cast to strict SkillEntry TypedDict ─────────────────────────
    return _cast_to_skill_entries(jd_skills)


def _merge_and_deduplicate(
    explicit: List[dict], implicit: List[dict]
) -> List[dict]:
    """
    Merge SkillNER (explicit) and JobBERT (implicit) results into a single
    deduplicated list keyed by taxonomy_id.

    Explicit skills from SkillNER have higher confidence weight.
    Implicit skills from JobBERT are included with lower confidence.

    Returns list of dicts with:
        {"taxonomy_id": str, "taxonomy_source": str, "label": str, "confidence_score": float}
    """
    seen: Set[str] = set()
    merged: List[dict] = []

    # SkillNER results — all have EMSI IDs, confidence = 0.7
    for skill in explicit:
        skill_id = skill.get("skill_id", "").strip()
        if skill_id and skill_id not in seen:
            seen.add(skill_id)
            merged.append({
                "taxonomy_id": skill_id,
                "taxonomy_source": "emsi",
                "label": skill.get("label", "").strip(),
                "confidence_score": 0.7,
            })

    # JobBERT results — may be EMSI-mapped or inferred
    for skill in implicit:
        skill_id = skill.get("skill_id", "").strip()
        source   = skill.get("source", "inferred")

        if skill_id and skill_id not in seen:
            seen.add(skill_id)

            if source == "emsi":
                # Hard skill mapped to EMSI via SkillNER — lower weight since JobBERT-sourced
                confidence = 0.3
            else:
                # Soft/contextual skill — lowest confidence tier
                confidence = 0.2

            merged.append({
                "taxonomy_id": skill_id,
                "taxonomy_source": source,
                "label": skill.get("label", "").strip(),
                "confidence_score": confidence,
            })

    return merged


def _cast_to_skill_entries(scored: List[dict]) -> List[SkillEntry]:
    entries: List[SkillEntry] = []
    for item in scored:
        try:
            tax_id  = str(item.get("taxonomy_id", "")).strip()
            source  = str(item.get("taxonomy_source", "emsi")).strip()
            label   = str(item.get("label", "")).strip()
            mastery = float(item.get("mastery_score", 0.3))
            mastery = max(0.0, min(1.0, mastery))
            conf    = float(item.get("confidence_score", 0.5))
            conf    = max(0.0, min(1.0, conf))

            if not tax_id or not label:
                continue

            entries.append({
                "taxonomy_id": tax_id,
                "taxonomy_source": source,
                "label": label,
                "mastery_score": mastery,
                "confidence_score": conf,
            })
        except (TypeError, ValueError):
            continue
    return entries