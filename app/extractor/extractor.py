import logging
from typing import Dict, List, Set

from app.extractor.schemas      import SkillEntry
from app.extractor.skillner_model import extract_explicit_skills
from app.extractor.jobbert_model  import extract_implicit_skills
from app.extractor.skillsim_model import map_to_esco
from app.extractor.groq_mastery   import estimate_mastery

logger = logging.getLogger(__name__)

# ── EXPANDED Dummy ESCO catalog (Fixes Bug 2) ──────────────────────────────────
_DUMMY_ESCO_CATALOG: Dict[str, dict] = {
    "http://data.europa.eu/esco/skill/ks-python"   : {"label": "Python (programming language)"},
    "http://data.europa.eu/esco/skill/ks-react"    : {"label": "React (JavaScript library)"},
    "http://data.europa.eu/esco/skill/ks-ml"       : {"label": "machine learning"},
    "http://data.europa.eu/esco/skill/ks-docker"   : {"label": "Docker"},
    "http://data.europa.eu/esco/skill/ks-restapi"  : {"label": "REST API development"},
    "http://data.europa.eu/esco/skill/ks-aws"      : {"label": "Amazon Web Services (AWS)"},
    "http://data.europa.eu/esco/skill/ks-sql"      : {"label": "SQL"},
    "http://data.europa.eu/esco/skill/ks-agile"    : {"label": "Agile software development"},
    # Added for Personas B, C, D, E
    "http://data.europa.eu/esco/skill/ks-scm"      : {"label": "supply chain management"},
    "http://data.europa.eu/esco/skill/ks-forklift" : {"label": "operate forklift"},
    "http://data.europa.eu/esco/skill/ks-seo"      : {"label": "search engine optimisation"},
    "http://data.europa.eu/esco/skill/ks-pandas"   : {"label": "pandas (Python)"},
    "http://data.europa.eu/esco/skill/ks-cust"     : {"label": "customer service"},
    "http://data.europa.eu/esco/skill/ks-payroll"  : {"label": "manage payroll"},
}

def extract_skills(raw_text: str) -> List[SkillEntry]:
    if not raw_text or not raw_text.strip():
        logger.warning("[Orchestrator] Empty input text. Returning [].")
        return []

    logger.info("[Orchestrator] Starting skill extraction pipeline.")

    # ── Stage 1: Explicit extraction (SkillNER) ──────────────────────────────
    explicit_skills: List[str] = []
    try:
        explicit_skills = extract_explicit_skills(raw_text)
    except Exception as exc:
        logger.error("[Orchestrator] Stage 1 failed: %s", exc)

    # ── Stage 2: Implicit extraction (JobBERT) ───────────────────────────────
    implicit_skills: List[str] = []
    try:
        implicit_skills = extract_implicit_skills(raw_text)
    except Exception as exc:
        logger.error("[Orchestrator] Stage 2 failed: %s", exc)

    # ── Stage 3: Merge and deduplicate ───────────────────────────────────────
    merged = _merge_and_deduplicate(explicit_skills, implicit_skills)

    if not merged:
        return []

    # ── Stage 4: Semantic ESCO mapping (SkillSim) ────────────────────────────
    mapped_skills: List[dict] = []
    try:
        mapped_skills = map_to_esco(merged, _DUMMY_ESCO_CATALOG)
    except Exception as exc:
        logger.error("[Orchestrator] Stage 4 failed: %s", exc)

    if not mapped_skills:
        return []
        
    # [BUG 3 FIX]: Deduplicate by ESCO URI so "AWS" and "Amazon Web Services" don't double up
    unique_mapped = {}
    for skill in mapped_skills:
        uri = skill["esco_uri"]
        if uri not in unique_mapped:
            unique_mapped[uri] = skill
    mapped_skills = list(unique_mapped.values())

    # ── Stage 5: Mastery scoring (Groq / Llama-3) ────────────────────────────
    scored_skills: List[dict] = []
    try:
        scored_skills = estimate_mastery(raw_text, mapped_skills)
    except Exception as exc:
        logger.error("[Orchestrator] Stage 5 failed: %s", exc)
        scored_skills = [{"esco_uri": s["esco_uri"], "label": s["label"], "mastery_score": 0.3} for s in mapped_skills]

    # ── Stage 6: Cast to strict SkillEntry TypedDict ─────────────────────────
    return _cast_to_skill_entries(scored_skills)

def _merge_and_deduplicate(explicit: List[str], implicit: List[str]) -> List[str]:
    seen: Set[str] = set()
    merged: List[str] = []
    for skill in explicit + implicit:
        normalized = skill.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            merged.append(normalized)
    return merged

def _cast_to_skill_entries(scored: List[dict]) -> List[SkillEntry]:
    entries: List[SkillEntry] = []
    for item in scored:
        try:
            uri   = str(item.get("esco_uri", "")).strip()
            label = str(item.get("label", "")).strip()
            score = float(item.get("mastery_score", 0.3))
            score = max(0.0, min(1.0, score))

            if not uri or not label:
                continue

            entries.append({"esco_uri": uri, "label": label, "mastery_score": score})
        except (TypeError, ValueError):
            continue
    return entries