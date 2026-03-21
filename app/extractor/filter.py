# app/extractor/filter.py
"""
LLM-powered post-extraction skill filter.

Purpose:
    SkillNER and JobBERT are recall-optimised — they over-extract by design.
    After merging their outputs, the combined list routinely contains:

      - Generic English verbs extracted as skills  ("manage", "write", "track")
      - Adjectives and adverbs                     ("scalable", "professional", "nice")
      - Company/brand names picked up as noun spans ("Blue Dart", "Samsung", "Jira")
      - Resume/JD boilerplate phrases              ("team player", "fast learner")
      - Fragments and substrings                   ("ad", "dec", "com", "lti")
      - Role titles misread as skills              ("operations manager", "floor supervisor")
      - Vague soft skills too generic to drive
        course routing                             ("problem solving", "leadership",
                                                    "communication")

    The static NOISE_TAXONOMY_IDS and NOISE_LABELS blocklists in gap_analyzer.py
    catch *known* noise by ID and label. This module catches *novel* noise —
    things neither list has seen before — by asking an LLM to reason about each
    skill in the context of the source document.

Design choices:
    - Single Groq call per filter pass: all candidates batched into one prompt
      rather than one call per skill. This keeps latency low and cost negligible.
    - Model: llama-3.1-8b-instant — same as groq_mastery.py. Fast, cheap, and
      more than capable for binary classification of short skill labels.
    - Fallback: on any API failure the full input list is returned unchanged.
      The filter is a quality improvement, not a hard dependency. The pipeline
      must never crash because the filter timed out.
    - Context-aware: both the skill list AND a truncated excerpt of the source
      text are sent. This lets the model distinguish "Dart" the language from
      "Blue Dart" the courier company, and "integration" the skill from the
      boilerplate phrase "successful integration of stakeholders".

Input / Output contract:
    filter_extracted_skills(
        skills   : List[dict],   # merged output from _merge_and_deduplicate()
        raw_text : str,          # the resume or JD text the skills were drawn from
        source   : str,          # "resume" | "jd"  — affects prompt framing
    ) -> List[dict]              # same schema, noise entries removed

    Each dict has at minimum: {"taxonomy_id", "taxonomy_source", "label"}
    The filter never modifies any field — it only decides keep vs discard.
"""

import json
import logging
import os
from typing import List, Dict, Any

from groq import Groq

logger = logging.getLogger(__name__)

_GROQ_MODEL  = "llama-3.1-8b-instant"
_GROQ_CLIENT = None

# Maximum characters of source text sent as context.
# Enough to give the model structural context without blowing the token budget.
_CONTEXT_CHARS = 3000


def _get_groq_client() -> Groq:
    global _GROQ_CLIENT
    if _GROQ_CLIENT is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "[SkillFilter] GROQ_API_KEY is not set. "
                "Cannot run LLM filter — returning input unchanged."
            )
        _GROQ_CLIENT = Groq(api_key=api_key)
    return _GROQ_CLIENT


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a precision skill-extraction auditor for an AI-powered corporate onboarding engine.

Your job is to review a list of skill candidates that were auto-extracted from a {source_type} \
and decide which ones are genuine, actionable skills that belong in a professional skills taxonomy, \
and which ones are extraction noise that must be discarded.

A skill is VALID if it is:
  - A concrete technical tool, language, framework, or platform
    (e.g. "Python", "React.js", "Docker", "PostgreSQL", "SAP ERP")
  - A well-defined professional methodology or practice
    (e.g. "Agile", "CI/CD", "Six Sigma", "Scrum", "Test-Driven Development")
  - A specific domain competency with a clear learning path
    (e.g. "supply chain management", "financial modelling", "machine learning",
     "network security", "data visualisation")
  - A certification or qualification with industry recognition
    (e.g. "AWS Certified Solutions Architect", "OSHA 30", "PMP")

A skill is NOISE and must be discarded if it is ANY of the following:
  1. A generic verb or action word — "manage", "write", "track", "read",
     "collaborate", "coordinate", "reach", "scale", "integrate", "execute"
  2. An adjective or adverb — "scalable", "professional", "nice",
     "high performance", "best practices", "detail oriented"
  3. A company name, brand, or product name that is NOT itself a learnable
     skill platform — "Blue Dart", "Samsung", "Google", "Infosys", "Jira"
     (NOTE: "Jira" IS a skill; "Samsung" is NOT)
  4. A role/job title — "operations manager", "floor supervisor",
     "team leader", "project manager" (titles are not skills)
  5. Resume/JD boilerplate soft-skill filler — "team player", "fast learner",
     "self-starter", "go-getter", "passionate about", "results-driven"
  6. Vague soft skills too generic to map to a course or certification —
     "communication", "leadership", "problem solving", "teamwork",
     "time management", "interpersonal skills", "flexibility",
     "critical thinking", "decision making", "adaptability"
     EXCEPTION: keep soft skills that have specific industry certifications
     or structured learning paths (e.g. "conflict resolution" in HR contexts)
  7. A word fragment or partial span — "ad", "dec", "com", "lti",
     "dart" (when the source text contains "Blue Dart Express"),
     "server side" when it is clearly a modifier not a skill
  8. Generic technology categories too vague for course mapping —
     "software", "technology", "system", "platform", "tool", "database",
     "cloud", "backend", "frontend", "web", "mobile", "API"
     EXCEPTION: keep these if they have a specific qualifier that makes them
     concrete, e.g. "cloud" → NOISE, "AWS cloud infrastructure" → VALID
  9. A business process or organisational term, not a learnable skill —
     "budget", "revenue", "operations", "workflow", "integration",
     "planning", "strategy", "management" (when standalone/generic)

SOURCE DOCUMENT EXCERPT:
{source_excerpt}

CANDIDATE SKILLS TO AUDIT:
{skill_list_json}

TASK:
For each skill in the list, return "keep" or "discard" with a one-sentence reason.

CRITICAL OUTPUT FORMAT RULES:
- Return ONLY a valid JSON array. No markdown. No preamble. No explanation outside the JSON.
- The array must have EXACTLY the same number of elements as the input list, in the same order.
- Each element must be an object with exactly three keys:
    "label"  : the skill label exactly as given in the input
    "verdict": either "keep" or "discard"  (lowercase, no other values)
    "reason" : one sentence (max 15 words) explaining your decision

Example output for a 3-item input:
[
  {{"label": "Python", "verdict": "keep", "reason": "Concrete programming language with a clear learning path."}},
  {{"label": "manage", "verdict": "discard", "reason": "Generic verb, not a learnable skill."}},
  {{"label": "Blue Dart", "verdict": "discard", "reason": "Courier company name, not a skill."}}
]
"""


def filter_extracted_skills(
    skills: List[Dict[str, Any]],
    raw_text: str,
    source: str = "resume",
) -> List[Dict[str, Any]]:
    """
    Filter a merged skill list using an LLM to remove extraction noise.

    This runs AFTER _merge_and_deduplicate() and BEFORE mastery scoring
    (resume) or the JD mastery placeholder assignment.

    Args:
        skills:   List of merged skill dicts. Each must have at least
                  {"taxonomy_id": str, "taxonomy_source": str, "label": str}.
        raw_text: The original resume or JD text (used as context for the LLM).
        source:   "resume" or "jd" — adjusts prompt framing.

    Returns:
        Filtered list with noise entries removed. Preserves the original dict
        structure and field values for all kept entries.
        On any failure, returns the original `skills` list unchanged.
    """
    if not skills:
        return skills

    source_type  = "resume" if source == "resume" else "job description"
    # Truncate context — enough for structural understanding, not the whole doc
    source_excerpt = raw_text.strip()[:_CONTEXT_CHARS]
    if len(raw_text.strip()) > _CONTEXT_CHARS:
        source_excerpt += "\n[... truncated for brevity ...]"

    # Build the compact label list for the prompt
    # Only send label — taxonomy_id is internal plumbing, not meaningful to the LLM
    skill_list_for_prompt = [{"label": s.get("label", "")} for s in skills]

    prompt = _SYSTEM_PROMPT.format(
        source_type    = source_type,
        source_excerpt = source_excerpt,
        skill_list_json = json.dumps(skill_list_for_prompt, indent=2),
    )

    try:
        client   = _get_groq_client()
        response = client.chat.completions.create(
            model       = _GROQ_MODEL,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.0,          # Deterministic — classification, not generation
            max_tokens  = 2048,
            # No response_format=json_object here because the model must return
            # an array, not an object. We parse manually with strip+fallback.
        )

        raw_output = response.choices[0].message.content.strip()

        # Strip markdown fences if the model wrapped the JSON anyway
        if raw_output.startswith("```"):
            raw_output = raw_output.split("```")[1]
            if raw_output.startswith("json"):
                raw_output = raw_output[4:]
            raw_output = raw_output.strip()

        verdicts: List[Dict[str, str]] = json.loads(raw_output)

        # Validate response length matches input
        if len(verdicts) != len(skills):
            logger.warning(
                "[SkillFilter] LLM returned %d verdicts for %d skills — "
                "length mismatch, returning input unchanged.",
                len(verdicts), len(skills),
            )
            return skills

        # Build label → verdict map for fast lookup
        verdict_map: Dict[str, str] = {}
        for item in verdicts:
            label   = item.get("label", "").strip().lower()
            verdict = item.get("verdict", "keep").strip().lower()
            reason  = item.get("reason", "")
            verdict_map[label] = verdict
            if verdict == "discard":
                logger.info(
                    "[SkillFilter] DISCARDED '%s' — %s", item.get("label", ""), reason
                )
            else:
                logger.debug(
                    "[SkillFilter] Kept '%s' — %s", item.get("label", ""), reason
                )

        # Filter the original skill dicts using the verdict map
        kept = [
            s for s in skills
            if verdict_map.get(s.get("label", "").strip().lower(), "keep") == "keep"
        ]

        removed_count = len(skills) - len(kept)
        logger.info(
            "[SkillFilter] Filter complete: %d kept, %d discarded from %d total (%s).",
            len(kept), removed_count, len(skills), source_type,
        )
        return kept

    except EnvironmentError as env_err:
        # No API key — log and pass through silently
        logger.warning("[SkillFilter] %s. Skipping filter.", env_err)
        return skills

    except json.JSONDecodeError as parse_err:
        logger.error(
            "[SkillFilter] Failed to parse LLM response as JSON: %s. "
            "Returning input unchanged.", parse_err,
        )
        return skills

    except Exception as exc:
        logger.error(
            "[SkillFilter] Unexpected error during filtering: %s. "
            "Returning input unchanged.", exc, exc_info=True,
        )
        return skills
