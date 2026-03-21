# app/extractor/skillner_model.py
"""
Explicit skill extraction using SkillNER + spaCy en_core_web_lg.

Extracts surface-form skill strings AND their EMSI taxonomy IDs from
candidate text via lexical matching against the EMSI SKILL_DB taxonomy.

CRITICAL CONSTRAINT (Phase 1 finding):
    The 'score' and 'len' fields in SkillNER output are np.int64 objects.
    They are intentionally NOT returned from this module to prevent fatal
    json.dumps() serialization errors downstream.
    Only clean Python str values are returned.

EMSI ID NOTE:
    SkillNER's SKILL_DB uses EMSI IDs as keys (e.g. "KS440L566SHJ6KQKFHKF").
    Match results include a 'skill_id' field containing the EMSI ID, sometimes
    with suffixes like '_fullUni', '_lowSurf', '_oneToken' — we strip these
    to get the clean EMSI ID.
"""

import logging
from typing import List, Optional, Dict

import spacy
from skillNer.skill_extractor_class import SkillExtractor
from skillNer.general_params import SKILL_DB
from spacy.matcher import PhraseMatcher

logger = logging.getLogger(__name__)

# Module-level singleton — spaCy and SkillNER are expensive to initialize.
# Loaded once on first call, reused for the lifetime of the process.
_extractor: Optional[SkillExtractor] = None
_nlp      : Optional[spacy.Language] = None

# Suffixes that SkillNER appends to skill IDs for different match types
_SKILL_ID_SUFFIXES = ("_fullUni", "_lowSurf", "_oneToken")

# Company names and proper nouns that commonly appear in resumes/JDs
# These should be excluded from skill extraction to avoid false positives
_COMPANY_NAMES_KEYWORDS = {
    "blue dart", "samsung", "google", "amazon", "microsoft", "apple",
    "ibm", "oracle", "salesforce", "github", "gitlab", "jira",
    "atlassian", "netflix", "uber", "lyft", "airbnb", "tesla",
    "accenture", "cognizant", "infosys", "tata", "wipro", "hcl",
    "cisco", "vmware", "dell", "hp", "lenovo", "intel", "nvidia",
    "qualcomm", "broadcom", "amd", "arm", "qualcomm", "analog",
    "texas instruments", "stm microelectronics", "nxp",
}


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


def _is_part_of_company_name(label: str, context_text: str) -> bool:
    """
    Check if a skill label is part of a company/proper noun in the context.
    
    For example, "dart" in "Blue Dart Express" should be filtered out.
    
    Args:
        label: The skill label to check (e.g., "dart").
        context_text: The surrounding text from the resume/JD.
        
    Returns:
        True if the label appears to be part of a company name, False otherwise.
    """
    label_lower = label.lower().strip()
    context_lower = context_text.lower()
    
    # Check if label appears in any company name
    for company in _COMPANY_NAMES_KEYWORDS:
        if label_lower in company and company in context_lower:
            logger.debug("[SkillNER] Filtered '%s' — detected as part of company name '%s'",
                        label, company)
            return True
    
    # Check for common company name patterns around the match
    # e.g., "Blue Dart" pattern with capitalized proper nouns
    idx = context_lower.find(label_lower)
    if idx >= 0:
        # Look at surrounding context (30 chars before and after)
        start = max(0, idx - 30)
        end = min(len(context_text), idx + len(label_lower) + 30)
        surrounding = context_text[start:end]
        
        # If surrounded by capitalized words, likely a proper noun
        # Simple heuristic: look for "Word Label Express" pattern
        words_before = surrounding[:surrounding.lower().find(label_lower)].split()
        words_after = surrounding[surrounding.lower().find(label_lower) + len(label_lower):].split()
        
        # If there are capitalized words directly adjacent, it might be a company name
        if words_before and words_before[-1] and words_before[-1][0].isupper():
            if words_after and words_after[0] and words_after[0][0].isupper():
                logger.debug("[SkillNER] Filtered '%s' — appears in proper noun pattern", label)
                return True
    
    return False


def _clean_skill_id(raw_id: str) -> str:
    """
    Strip SkillNER match-type suffixes from skill IDs.
    e.g. "KS440L566SHJ6KQKFHKF_fullUni" → "KS440L566SHJ6KQKFHKF"
    """
    for suffix in _SKILL_ID_SUFFIXES:
        if raw_id.endswith(suffix):
            return raw_id[: -len(suffix)]
    return raw_id


def extract_explicit_skills(text: str) -> List[Dict[str, str]]:
    """
    Extract explicitly mentioned skills from free-form resume or profile text.

    Parses both 'full_matches' (exact phrase matches) and 'ngram_scored'
    (surface-form n-gram matches) from SkillNER's annotation output.
    
    Filters out skills that appear to be part of company names or proper nouns.

    Args:
        text: Raw resume or profile text string.

    Returns:
        Deduplicated list of skill dicts, each with:
            {"skill_id": str, "label": str}
        e.g. [{"skill_id": "KS440L566SHJ6KQKFHKF", "label": "React.js"}, ...]

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

        seen  : set                  = set()
        skills: List[Dict[str, str]] = []

        # Process full_matches first — highest precision
        for match in results.get("full_matches", []):
            _add_skill(match, seen, skills, source="full_matches", context_text=text)

        # Process ngram_scored — broader recall
        for match in results.get("ngram_scored", []):
            _add_skill(match, seen, skills, source="ngram_scored", context_text=text)

        logger.info("[SkillNER] Extracted %d unique skills from text.", len(skills))
        return skills

    except Exception as exc:
        logger.error("[SkillNER] Extraction failed: %s", exc, exc_info=True)
        return []


def map_span_to_emsi(span: str) -> Optional[Dict[str, str]]:
    """
    Attempt to map a raw text span (e.g. from JobBERT) to an EMSI skill entry
    by running it through SkillNER's PhraseMatcher.

    Only returns a match if the EMSI skill has substantial token overlap with
    the input span. This prevents generic single words like "managed" or "ad"
    from being falsely mapped to unrelated EMSI entries.

    Args:
        span: A raw skill phrase string to attempt matching.

    Returns:
        {"skill_id": str, "label": str} if a high-quality match is found,
        else None.
    """
    if not span or not span.strip():
        return None

    # Reject very short spans (1-2 chars) — too ambiguous
    cleaned = span.strip().lower()
    if len(cleaned) <= 2:
        return None

    try:
        extractor   = _get_extractor()
        annotations = extractor.annotate(span)
        results     = annotations.get("results", {})

        # Try full_matches first (highest confidence)
        for match in results.get("full_matches", []):
            skill_id = _extract_skill_id(match)
            label    = _extract_label(match, skill_id)
            if skill_id and label and _is_quality_match(cleaned, label):
                return {"skill_id": skill_id, "label": label}

        # Fall back to ngram_scored with stricter threshold
        for match in results.get("ngram_scored", []):
            score = match.get("score", 0)
            if isinstance(score, (int, float)) and score >= 0.8:  # stricter for JobBERT remapping
                skill_id = _extract_skill_id(match)
                label    = _extract_label(match, skill_id)
                if skill_id and label and _is_quality_match(cleaned, label):
                    return {"skill_id": skill_id, "label": label}

        return None

    except Exception as exc:
        logger.debug("[SkillNER] map_span_to_emsi failed for '%s': %s", span, exc)
        return None


def _is_quality_match(span: str, emsi_label: str) -> bool:
    """
    Validate that the EMSI match is a genuine skill match, not a spurious
    single-token coincidence.

    Rules:
    1. Single-word spans that are common English words (not technical terms)
       are rejected unless the EMSI label is an exact match.
    2. Multi-word spans require >= 50% token overlap with the EMSI label.
    """
    span_tokens = set(span.lower().split())
    label_tokens = set(emsi_label.lower().split())

    # If the span is a single common word, require exact label match
    if len(span_tokens) == 1:
        # Single-word span must match the EMSI label closely
        # (label can be longer, e.g. span="sql" → label="sql" is fine)
        return span.lower().strip() in emsi_label.lower()

    # Multi-word spans: require meaningful overlap
    overlap = span_tokens & label_tokens
    # At least 50% of the span's tokens must appear in the EMSI label
    if len(span_tokens) > 0 and len(overlap) / len(span_tokens) >= 0.5:
        return True

    return False


def _extract_skill_id(match: dict) -> str:
    """Extract and clean the EMSI skill ID from a SkillNER match entry."""
    raw_id = match.get("skill_id", "")
    if not isinstance(raw_id, str):
        raw_id = str(raw_id)
    return _clean_skill_id(raw_id.strip())


def _extract_label(match: dict, skill_id: str) -> str:
    """
    Extract the human-readable label for a skill.
    Tries doc_node_value first, then falls back to SKILL_DB lookup.
    """
    label = match.get("doc_node_value", "")
    if not isinstance(label, str):
        label = str(label)
    label = label.strip()

    # If doc_node_value is empty, try SKILL_DB lookup
    if not label and skill_id in SKILL_DB:
        label = SKILL_DB[skill_id].get("skill_name", "")

    return label


def _add_skill(
    match : dict,
    seen  : set,
    skills: List[Dict[str, str]],
    source: str,
    context_text: str = ""
) -> None:
    """
    Extract the EMSI skill_id and doc_node_value from a match entry and
    add it to the deduplicated skills list.
    
    Filters out skills that appear to be part of company names.

    INTENTIONALLY ignores 'score' and 'len' — both are np.int64 and
    will cause json.dumps() to raise TypeError downstream.
    """
    skill_id = _extract_skill_id(match)
    label    = _extract_label(match, skill_id)

    if not skill_id or not label:
        return
    
    # Filter out skills that are part of company names
    if context_text and _is_part_of_company_name(label, context_text):
        logger.debug("[SkillNER] Filtered skill '%s' — appears in company name", label)
        return

    if skill_id not in seen:
        seen.add(skill_id)
        skills.append({"skill_id": skill_id, "label": label.lower()})
        logger.debug("[SkillNER] [%s] Found skill: id='%s' label='%s'", source, skill_id, label)