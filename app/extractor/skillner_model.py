# app/extractor/skillner_model.py
"""
Explicit skill extraction using SkillNER + spaCy en_core_web_lg.

Extracts surface-form skill strings from candidate text via lexical
matching against the ESCO/LinkedIn SKILL_DB taxonomy.

CRITICAL CONSTRAINT (Phase 1 finding):
    The 'score' and 'len' fields in SkillNER output are np.int64 objects.
    They are intentionally NOT returned from this module to prevent fatal
    json.dumps() serialization errors downstream.
    Only clean Python str values are returned.
"""

import logging
from typing import List, Optional

import spacy
#from skillNer import SkillExtractor
from skillNer.skill_extractor_class import SkillExtractor
from skillNer.general_params import SKILL_DB
from spacy.matcher import PhraseMatcher

logger = logging.getLogger(__name__)

# Module-level singleton — spaCy and SkillNER are expensive to initialize.
# Loaded once on first call, reused for the lifetime of the process.
_extractor: Optional[SkillExtractor] = None
_nlp       : Optional[spacy.Language] = None


def _get_extractor() -> SkillExtractor:
    """
    Lazy-initialize and cache the SkillNER extractor singleton.
    Thread-safety note: acceptable for single-worker Uvicorn; add a lock
    if moving to multi-threaded workers.
    """
    global _extractor, _nlp
    if _extractor is None:
        logger.info("[SkillNER] Loading spaCy en_core_web_lg...")
        _nlp = spacy.load("en_core_web_lg")
        logger.info("[SkillNER] Initializing SkillExtractor with SKILL_DB...")
        _extractor = SkillExtractor(_nlp, SKILL_DB, PhraseMatcher)
        logger.info("[SkillNER] Ready.")
    return _extractor


def extract_explicit_skills(text: str) -> List[str]:
    """
    Extract explicitly mentioned skills from free-form resume or profile text.

    Parses both 'full_matches' (exact phrase matches) and 'ngram_scored'
    (surface-form n-gram matches) from SkillNER's annotation output.

    Args:
        text: Raw resume or profile text string.

    Returns:
        Deduplicated list of skill surface-form strings (lowercase).
        e.g. ["python", "react", "rest apis", "docker"]

        Returns empty list on empty input or extraction failure.

    Raises:
        Does NOT raise — all exceptions are caught and logged.
        Downstream orchestrator receives [] on failure, not a crash.
    """
    if not text or not text.strip():
        logger.warning("[SkillNER] Received empty text, returning [].")
        return []

    try:
        extractor   = _get_extractor()
        annotations = extractor.annotate(text)
        results     = annotations.get("results", {})

        seen  : set       = set()
        skills: List[str] = []

        # Process full_matches first — highest precision
        for match in results.get("full_matches", []):
            _add_skill(match, seen, skills, source="full_matches")

        # Process ngram_scored — broader recall
        for match in results.get("ngram_scored", []):
            _add_skill(match, seen, skills, source="ngram_scored")

        logger.info("[SkillNER] Extracted %d unique skills from text.", len(skills))
        return skills

    except Exception as exc:
        logger.error("[SkillNER] Extraction failed: %s", exc, exc_info=True)
        return []


def _add_skill(
    match : dict,
    seen  : set,
    skills: List[str],
    source: str
) -> None:
    """
    Extract ONLY the doc_node_value string from a match entry and
    add it to the deduplicated skills list.

    INTENTIONALLY ignores 'score' and 'len' — both are np.int64 and
    will cause json.dumps() to raise TypeError downstream.
    """
    raw_value = match.get("doc_node_value", "")

    # Defensive: ensure we always store a clean Python str
    if not isinstance(raw_value, str):
        raw_value = str(raw_value)

    surface = raw_value.strip().lower()

    if surface and surface not in seen:
        seen.add(surface)
        skills.append(surface)
        logger.debug("[SkillNER] [%s] Found skill: '%s'", source, surface)