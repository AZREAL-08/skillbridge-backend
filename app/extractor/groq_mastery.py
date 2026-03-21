"""
Profile-classification mastery scoring.

Architecture:
  1. Single Groq call classifies the candidate profile type
  2. Bloom-level decay formula computes mastery per skill
  3. Mention boost for skills explicitly listed in the resume
  4. Cap/floor at 0.10–0.95

This replaces the old per-skill LLM mastery scoring. One Groq call per
resume instead of N, consistent scores, and no dependency on per-skill
context sentences.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from groq import Groq

logger = logging.getLogger(__name__)

_GROQ_MODEL = "llama-3.3-70b-versatile"
_GROQ_CLIENT = None


def _get_groq_client() -> Groq:
    global _GROQ_CLIENT
    if _GROQ_CLIENT is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "[GroqMastery] GROQ_API_KEY environment variable is not set."
            )
        _GROQ_CLIENT = Groq(api_key=api_key)
    return _GROQ_CLIENT


# ── Profile Classification Constants ─────────────────────────────────────────

PROFILE_BASE_SCORES: Dict[str, float] = {
    "senior_professional": 0.85,   # 5+ years, leadership roles, production systems
    "mid_professional":    0.70,   # 2-5 years, independent contributor
    "junior_professional": 0.55,   # 0-2 years, some production experience
    "hackathon_builder":   0.60,   # student/fresher but ships projects, competitions
    "fresher_academic":    0.35,   # degree only, no meaningful projects
}

BLOOM_DECAY: Dict[int, float] = {
    1: 1.00,   # Foundational — full base score
    2: 0.95,   # Core concepts — nearly full
    3: 0.80,   # Applied — moderate decay
    4: 0.60,   # Advanced — significant decay
    5: 0.40,   # Expert — steep decay
    6: 0.25,   # Creative/Architectural — assume not mastered
}

MENTION_BOOST: float = 0.15   # Boost for skills explicitly mentioned in resume
PROJECT_BOOST: float = 0.10   # Boost for skills used in described projects

_PROFILE_CLASSIFY_PROMPT = """
You are a resume classification engine. Your sole task is to classify the candidate into exactly one category based on their overall profile.

Categories:
- "senior_professional": 5+ years experience, led teams, architected production systems, mentored others
- "mid_professional": 2-5 years experience, independent contributor, shipped to production
- "junior_professional": 0-2 years professional experience, some real work, entry-level roles
- "hackathon_builder": student or fresher who actively ships projects, competes in hackathons, has a portfolio
- "fresher_academic": degree only, minimal or no projects, no competitions, no work experience

RULES:
1. Look at the OVERALL picture: years of experience, role titles, project complexity, leadership signals, competition wins.
2. A student with 5+ hackathons and shipped projects is "hackathon_builder", NOT "fresher_academic".
3. Someone with "Lead" or "Senior" in their title AND 5+ years is "senior_professional".
4. Return ONLY valid JSON. No explanation outside the JSON.

OUTPUT FORMAT:
{"profile_type": "<category>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}
""".strip()


# ── Core API ──────────────────────────────────────────────────────────────────


def classify_profile(resume_text: str) -> Tuple[str, float, str]:
    """
    Single Groq call to classify the candidate's profile type.

    Returns:
        (profile_type, base_score, reasoning)
    """
    if not resume_text or not resume_text.strip():
        logger.warning("[GroqMastery] Empty resume text, defaulting to fresher_academic.")
        return "fresher_academic", PROFILE_BASE_SCORES["fresher_academic"], "Empty resume"

    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=_GROQ_MODEL,
            temperature=0.0,
            max_tokens=256,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PROFILE_CLASSIFY_PROMPT},
                {"role": "user", "content": f"RESUME:\n{resume_text.strip()}"},
            ],
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        profile_type = parsed.get("profile_type", "junior_professional")
        reasoning = parsed.get("reasoning", "")

        if profile_type not in PROFILE_BASE_SCORES:
            logger.warning(
                "[GroqMastery] Unknown profile type '%s', defaulting to junior_professional.",
                profile_type,
            )
            profile_type = "junior_professional"

        base_score = PROFILE_BASE_SCORES[profile_type]
        logger.info(
            "[GroqMastery] Classified as '%s' (base=%.2f). Reasoning: %s",
            profile_type,
            base_score,
            reasoning,
        )
        return profile_type, base_score, reasoning

    except Exception as exc:
        logger.error("[GroqMastery] Profile classification failed: %s", exc)
        return "junior_professional", PROFILE_BASE_SCORES["junior_professional"], "Classification failed, using fallback"


def compute_mastery_scores(
    mapped_skills: List[dict],
    resume_text: str,
    catalog: List[dict],
) -> List[dict]:
    """
    Compute mastery scores using profile classification + bloom decay + mention boost.

    Args:
        mapped_skills: Merged/deduped skills with taxonomy_id, label, etc.
        resume_text:   Raw resume text for profile classification and mention detection.
        catalog:       Full course catalog for bloom level lookup.

    Returns:
        List of skill dicts with mastery_score populated.
    """
    if not mapped_skills:
        return []

    # Step 1: Classify the candidate profile (single Groq call)
    profile_type, base_score, reasoning = classify_profile(resume_text)

    # Step 2: Build bloom-level lookup from catalog
    # Maps taxonomy_id → lowest bloom_level among courses teaching that skill
    skill_bloom_map = _build_skill_bloom_map(catalog)

    # Step 3: Detect which skills are explicitly mentioned in the resume
    mentioned_ids = _find_mentioned_skills(mapped_skills, resume_text)

    # Step 4: Compute mastery for each skill
    scored: List[dict] = []
    for skill in mapped_skills:
        tax_id = skill.get("taxonomy_id", "")
        label = skill.get("label", "")

        bloom_level = skill_bloom_map.get(tax_id, 3)  # Default to applied level
        boost = MENTION_BOOST if tax_id in mentioned_ids else 0.0

        mastery = _final_mastery(base_score, bloom_level, boost)

        scored.append({
            "taxonomy_id": tax_id,
            "taxonomy_source": skill.get("taxonomy_source", "emsi"),
            "label": label,
            "mastery_score": mastery,
            "confidence_score": skill.get("confidence_score", 0.5),
            "profile_type": profile_type,
            "profile_reasoning": reasoning,
        })

    return scored


# ── Internal helpers ──────────────────────────────────────────────────────────


def _build_skill_bloom_map(catalog: List[dict]) -> Dict[str, int]:
    """
    Build a mapping from taxonomy_id → lowest bloom_level across all courses
    that teach that skill.

    Lower bloom = more foundational. If a skill appears in multiple courses,
    we use the lowest bloom level for mastery decay (most generous interpretation).
    """
    bloom_map: Dict[str, int] = {}
    for course in catalog:
        bloom = course.get("bloom_level", 3)
        for skill_id in course.get("skills_taught", []):
            if skill_id not in bloom_map or bloom < bloom_map[skill_id]:
                bloom_map[skill_id] = bloom
    return bloom_map


def _find_mentioned_skills(
    skills: List[dict], resume_text: str
) -> Set[str]:
    """
    Check which skills are explicitly mentioned in the resume text.
    Returns set of taxonomy_ids that are mentioned.

    Uses simple case-insensitive substring matching on the skill label.
    """
    mentioned: Set[str] = set()
    text_lower = resume_text.lower()

    for skill in skills:
        label = skill.get("label", "").lower().strip()
        if not label:
            continue

        # Check if the skill label appears in the resume text
        if label in text_lower:
            mentioned.add(skill.get("taxonomy_id", ""))
            continue

        # Also check common variations (e.g., "react js" → "react", "node js" → "node")
        base_label = label.split()[0] if " " in label else label
        if len(base_label) >= 3 and base_label in text_lower:
            mentioned.add(skill.get("taxonomy_id", ""))

    return mentioned


def _final_mastery(base_score: float, bloom_level: int, mention_boost: float) -> float:
    """
    Compute final mastery score with bloom decay and mention boost.

    Formula: clamp(base_score * BLOOM_DECAY[bloom] + mention_boost, 0.10, 0.95)
    """
    decay = BLOOM_DECAY.get(bloom_level, 0.80)
    raw = (base_score * decay) + mention_boost
    return round(max(0.10, min(0.95, raw)), 2)