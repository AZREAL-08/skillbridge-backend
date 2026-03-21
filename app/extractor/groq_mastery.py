"""
Profile-classification mastery scoring.

Architecture:
  1. Single Groq call classifies the candidate profile type
  2. Bloom-level decay formula computes mastery per skill
  3. Mention boost for skills explicitly listed in the resume
  4. Negation detection: hard-caps skills at 0.45 when resume says
     "not production", "no experience", "basic only", etc.
  5. Leadership/scale boost: +0.10 for skills used alongside metrics,
     "led team", "architected", "enterprise"
  6. Cap/floor at 0.10–0.95
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

MENTION_BOOST: float = 0.15      # Boost for skills explicitly mentioned in resume
LEADERSHIP_BOOST: float = 0.10   # Boost for skills with leadership/scale signals
CERTIFICATION_BOOST: float = 0.20 # Boost for certified skills
NEGATION_CAP: float = 0.45       # Hard cap when negation context detected
CERTIFICATION_FLOOR: float = 0.90 # Minimum mastery for certified skills
SENIORITY_FLOOR: float = 0.85     # Minimum mastery for 5+ years / daily core skills

# ── Negation Detection ────────────────────────────────────────────────────────
# Phrases that indicate the candidate does NOT have deep expertise in a skill.
# When any of these appear near a skill mention, mastery is hard-capped at 0.45.

NEGATION_PHRASES = [
    # Strong negations
    "not production",
    "no production",
    "not production-grade",
    "not production grade",
    "not for production",
    
    # Generic non-proficiency
    "not proficient",
    "no experience",
    "no real experience",
    "zero experience",
    "unfamiliar with",
    "not familiar",
    "no familiarity",
    "no experience with",
    
    # Time/scope bounded
    "never designed",
    "never built",
    "never used",
    "never worked",
    "never touched",
    "never developed",
    "never implemented",
    "never optimized",
    
    # Hands-on/professional
    "no hands-on",
    "no professional",
    "no work experience",
    "only theoretical",
    "read-only",
    "read only",
    "no schema design",
    
    # Level/capability
    "basic only",
    "basic understanding",
    "limited experience",
    "minimal experience",
    "beginner level",
    "junior level only",
    "basic joins",
    "basic select",
    
    # Not expert/advanced
    "not comfortable",
    "not expert",
    "no advanced",
    "not advanced",
    "no deep",
    "not deep",
    "surface level",
    
    # Tool-specific non-proficiency
    "no pivot tables",
    "no formal data analysis",
    "no power bi",
    
    # Scope/tool constraints
    "prototyping only",
    "small internal tools",
    "familiar with but",
    "data entry level only",
    "scripting only",
    
    # Weak signals
    "only familiar",
    "only learning",
    "only briefly",
    "only exposure",
    "first exposure",
    "struggled with",
    "weak in",
]

# ── Leadership/Scale Signal Detection ─────────────────────────────────────────
# Phrases that indicate the candidate has deep, real-world expertise.
# When these appear near a skill mention, an extra +0.10 boost is applied.

LEADERSHIP_PATTERNS = [
    # Team leadership
    r"led\s+(?:the\s+)?(?:team|frontend|backend|architecture|development|engineering|platform)",
    r"led\s+a\s+team\s+of\s+\d+",
    r"leading\s+(?:team|development|engineering)",
    r"managed\s+(?:a\s+)?team\s+of",
    
    # Architecture/system design
    r"architected",
    r"designed\s+(?:the\s+)?(?:system|architecture|platform|solution)",
    r"built\s+(?:the\s+)?(?:architecture|system|platform)",
    r"core\s+(?:system|architecture|component)",
    
    # Mentoring
    r"mentored\s+\d+",
    r"trained\s+\d+\+?",
    r"onboarded",
    
    # Scale metrics - users/customers
    r"\d+k\+?\s+(?:users|merchants|customers|transactions)",
    r"\d+\.?\d*m\+?\s+(?:users|merchants|customers|transactions)",
    r"million\+?\s+(?:users|customers|merchants)",
    r"thousands?\s+of\s+(?:users|customers|merchants)",
    
    # Business metrics - teams/adoption
    r"\d+\s+(?:product\s+)?teams",
    r"adopted\s+by\s+(?:\d+\s+)?(?:multiple|other|various)\s+(?:teams|products|services)",
    r"cross\s+(?:all|multiple|different)\s+(?:product|teams|services)",
    
    # Production scale indicators
    r"production(?:\s+.{0,30})?(?:system|scale|environment)",
    r"enterprise\s+(?:saas|application|platform|system|customer)",
    r"high\s+(?:throughput|traffic|concurrency|scale)",
    r"99\.\d+%\s+(?:accuracy|uptime|availability|reliability)",
    
    # Performance improvements
    r"(?:reduced|decreased|improved|increased)\s+.{0,30}\d+\s*%",
    r"zero\s+(?:incidents|downtime|bugs|critical)",
    
    # Scope indicators
    r"daily\s+(?:workflow|use|driver|tool)",
    r"across\s+(?:all|multiple|different|global)\s+(?:roles|teams|products|regions)",
    r"core\s+(?:skill|competency|technology)",
    r"primary\s+(?:skill|technology|tool)",
    
    # Promotion/recognition
    r"promoted\s+\d+\s+(?:times?|levels?)",
    r"promoted\s+to\s+(?:senior|lead|principal)",
    
    # Responsibility breadth
    r"owned\s+(?:the|full|complete|entire)\s+(?:system|platform|product|codebase)",
    r"responsible\s+for\s+(?:building|designing|architecting|leading)",
]

# ── Certification Detection ───────────────────────────────────────────────────
CERTIFICATION_PATTERNS = [
    r"certified",
    r"certification",
    r"certificate",
    r"osha\s+30-hour",
    r"forklift\s+certified",
]

# ── Seniority/Years-in-Role Detection ─────────────────────────────────────────
SENIORITY_PATTERNS = [
    r"5\+\s+years",
    r"8\+\s+years",
    r"10\+\s+years",
    r"decade\s+of",
    r"over\s+5\s+years",
    r"over\s+8\s+years",
    r"daily\s+driver",
    r"used\s+daily",
    r"years\s+of\s+experience",
    r"throughout\s+tenure",
]


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
        logger.info("[GroqMastery] Empty resume text, defaulting to fresher_academic.")
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
            logger.info(
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
    Compute mastery scores using profile classification + bloom decay + context signals.

    Pipeline per skill:
      1. base_score from profile classification
      2. × BLOOM_DECAY[bloom_level] from catalog
      3. + MENTION_BOOST if skill label found in resume
      4. + LEADERSHIP_BOOST if leadership/scale signals found near skill
      5. + CERTIFICATION_BOOST if certification signals found near skill
      6. cap at NEGATION_CAP (0.45) if negation phrases found near skill
      7. apply CERTIFICATION_FLOOR (0.90) or SENIORITY_FLOOR (0.85) if applicable
      8. clamp to [0.10, 0.95]
    """
    if not mapped_skills:
        return []

    # Step 1: Classify the candidate profile (single Groq call)
    profile_type, base_score, reasoning = classify_profile(resume_text)

    # Step 2: Build bloom-level lookup from catalog
    skill_bloom_map = _build_skill_bloom_map(catalog)

    # Step 3: Detect mention, negation, and leadership context per skill
    text_lower = resume_text.lower()

    scored: List[dict] = []
    for skill in mapped_skills:
        tax_id = skill.get("taxonomy_id", "")
        label = skill.get("label", "")

        bloom_level = skill_bloom_map.get(tax_id, 3)  # Default to applied level

        # Check if mentioned in resume
        is_mentioned = _is_skill_mentioned(label, text_lower)
        mention_boost = MENTION_BOOST if is_mentioned else 0.0

        # Check for negation context around this skill
        is_negated = _detect_negation_context(label, text_lower)

        # Check for leadership/scale signals around this skill
        has_leadership = _detect_leadership_signal(label, text_lower)
        leadership_boost = LEADERSHIP_BOOST if has_leadership and not is_negated else 0.0

        # Check for certification
        has_cert = _detect_certification(label, text_lower)
        cert_boost = CERTIFICATION_BOOST if has_cert and not is_negated else 0.0
        
        # Check for seniority
        has_seniority = _detect_seniority(label, text_lower)

        # Compute mastery
        mastery = _final_mastery(
            base_score, bloom_level, mention_boost, leadership_boost, cert_boost, 
            is_negated, has_cert, has_seniority
        )

        scored.append({
            "taxonomy_id": tax_id,
            "taxonomy_source": skill.get("taxonomy_source", "emsi"),
            "label": label,
            "mastery_score": mastery,
            "confidence_score": skill.get("confidence_score", 0.5),
            "profile_type": profile_type,
            "profile_reasoning": reasoning,
        })

        if is_negated:
            logger.debug(
                "[GroqMastery] NEGATION detected for '%s' — capped at %.2f", label, mastery
            )
        if has_leadership:
            logger.debug(
                "[GroqMastery] LEADERSHIP signal for '%s' — boosted", label
            )
        if has_cert:
            logger.debug(
                "[GroqMastery] CERTIFICATION detected for '%s' — boosted/floored", label
            )
        if has_seniority:
            logger.debug(
                "[GroqMastery] SENIORITY detected for '%s' — floored", label
            )

    return scored


# ── Internal helpers ──────────────────────────────────────────────────────────


def _build_skill_bloom_map(catalog: List[dict]) -> Dict[str, int]:
    """
    Build a mapping from taxonomy_id → lowest bloom_level across all courses
    that teach that skill.
    """
    bloom_map: Dict[str, int] = {}
    for course in catalog:
        bloom = course.get("bloom_level", 3)
        for skill_id in course.get("skills_taught", []):
            if skill_id not in bloom_map or bloom < bloom_map[skill_id]:
                bloom_map[skill_id] = bloom
    return bloom_map


def _is_skill_mentioned(label: str, text_lower: str) -> bool:
    """Check if a skill label appears in the resume text."""
    label_lower = label.lower().strip()
    if not label_lower:
        return False

    if label_lower in text_lower:
        return True

    # Check base word for multi-word labels (e.g., "react js" → "react")
    base_label = label_lower.split()[0] if " " in label_lower else label_lower
    if len(base_label) >= 3 and base_label in text_lower:
        return True

    return False


def _get_skill_context(label: str, text_lower: str, window: int = 400) -> str:
    """
    Extract the text window surrounding a skill mention in the resume.
    Returns the surrounding context (up to `window` chars on each side).
    Returns empty string if skill not found.
    
    Tries multiple strategies:
      1. Exact label match
      2. Base word match (first word from multi-word labels)
      3. Search for comma-separated variations
    """
    label_lower = label.lower().strip()
    if not label_lower:
        return ""

    # Try exact label first
    idx = text_lower.find(label_lower)

    # Try base word if exact not found
    if idx == -1 and " " in label_lower:
        base = label_lower.split()[0]
        if len(base) >= 3:
            idx = text_lower.find(base)

    # Try individual words separated by slashes/dots (e.g., "c++")
    if idx == -1 and ("+" in label_lower or "." in label_lower or "#" in label_lower):
        # For C++, C#, .NET etc, search for first meaningful part
        search_parts = re.split(r'[+.#]', label_lower)
        for part in search_parts:
            if len(part) >= 3:
                idx = text_lower.find(part)
                if idx != -1:
                    break

    if idx == -1:
        return ""

    start = max(0, idx - window)
    end = min(len(text_lower), idx + len(label_lower) + window)
    return text_lower[start:end]


def _detect_negation_context(label: str, text_lower: str) -> bool:
    """
    Check if a skill is mentioned with negation context in the resume.

    Scans a 60-char window around the skill mention for negation phrases
    like "not production", "no experience", "basic only", etc.
    
    Also checks for negation patterns that may precede the skill mention
    by looking at the full window (before and after).

    Returns True if negation detected — mastery should be hard-capped.
    """
    # Reduce window to 60 chars for negation to avoid capturing unrelated "no experience" phrases
    context = _get_skill_context(label, text_lower, window=60)
    if not context:
        return False

    # Check for direct phrase matches
    for phrase in NEGATION_PHRASES:
        if phrase.lower() in context:
            logger.debug(
                "[GroqMastery] Negation phrase '%s' found near '%s'", phrase, label
            )
            return True

    # Additional pattern check: look for "no/never/zero ... [skill]" patterns
    # This catches cases where negation precedes the skill
    skill_base = label.split()[0].lower() if " " in label else label.lower()
    negation_words = ["no ", "never ", "zero ", "unfamiliar ", "not "]
    
    for neg_word in negation_words:
        # Look for negation word followed by the skill base word somewhere in context
        # Using a much smaller max distance (e.g. 30 chars)
        pattern = f"{neg_word}.{{0,30}}{re.escape(skill_base)}"
        if re.search(pattern, context, re.DOTALL | re.IGNORECASE):
            logger.debug(
                "[GroqMastery] Negation pattern '%s.*%s' found in '%s'", neg_word, skill_base, label
            )
            return True

    return False


def _detect_leadership_signal(label: str, text_lower: str) -> bool:
    """
    Check if a skill is mentioned alongside leadership/scale signals.

    Scans a 150-char window around the skill mention for patterns like
    "led team of 5", "50k+ users", "enterprise SaaS", "architected", etc.

    Returns True if leadership signals found — skill gets a +0.10 boost.
    """
    context = _get_skill_context(label, text_lower, window=150)
    if not context:
        return False

    for pattern in LEADERSHIP_PATTERNS:
        if re.search(pattern, context, re.IGNORECASE):
            logger.debug(
                "[GroqMastery] Leadership signal '%s' found near '%s'", pattern, label
            )
            return True

    return False


def _detect_certification(label: str, text_lower: str) -> bool:
    """Check for certification phrases near the skill."""
    context = _get_skill_context(label, text_lower, window=100)
    if not context:
        return False
    for pattern in CERTIFICATION_PATTERNS:
        if re.search(pattern, context, re.IGNORECASE):
            return True
    return False


def _detect_seniority(label: str, text_lower: str) -> bool:
    """Check for seniority/years-in-role phrases in a larger context."""
    # Seniority signals like "8 years experience" might be in a header far from the skill
    context = _get_skill_context(label, text_lower, window=600)
    if not context:
        # If skill not found, check if seniority patterns exist ANYWHERE in resume 
        # as a fallback if the skill is known to be a core daily driver
        context = text_lower

    for pattern in SENIORITY_PATTERNS:
        if re.search(pattern, context, re.IGNORECASE):
            return True
    return False


def _final_mastery(
    base_score: float,
    bloom_level: int,
    mention_boost: float,
    leadership_boost: float,
    cert_boost: float,
    is_negated: bool,
    has_cert: bool,
    has_seniority: bool,
) -> float:
    """
    Compute final mastery score with bloom decay, boosts, and negation cap.

    Formula:
      raw = base_score × BLOOM_DECAY[bloom] + boosts
      if negated: raw = min(raw, NEGATION_CAP)
      if has_cert: raw = max(raw, CERTIFICATION_FLOOR)
      if has_seniority: raw = max(raw, SENIORITY_FLOOR)
      return clamp(raw, 0.10, 0.95)
    """
    decay = BLOOM_DECAY.get(bloom_level, 0.80)
    raw = (base_score * decay) + mention_boost + leadership_boost + cert_boost

    if is_negated:
        raw = min(raw, NEGATION_CAP)
    
    # Floors apply only if NOT negated
    if not is_negated:
        if has_cert:
            raw = max(raw, CERTIFICATION_FLOOR)
        if has_seniority:
            raw = max(raw, SENIORITY_FLOOR)

    return round(max(0.10, min(0.95, raw)), 2)
