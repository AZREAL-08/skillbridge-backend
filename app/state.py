"""
ALL TypedDicts live here — shared contract between backend modules.
Frontend mirrors these in lib/types.ts.
"""

from typing import TypedDict, List, Literal

class SkillEntry(TypedDict):
    esco_uri: str
    label: str
    mastery_score: float       # 0.0–1.0, from Groq mastery inference
    confidence_score: float    # 0.0–1.0, multi-model weighted confidence

class PathwayCourse(TypedDict):
    course_id: str
    node_state: Literal["skipped", "assigned", "prerequisite"]
    mastery_score: float       # User's current mastery of this course's skill
    confidence_score: float    # Drives UI dotted-border logic (< 0.70)

class Metrics(TypedDict):
    baseline_courses: int      # Always 30 (len of catalog)
    assigned_courses: int      # len(final_pathway where node_state != "skipped")
    reduction_pct: float       # ((baseline - assigned) / baseline) * 100

class CurrentState(TypedDict):
    raw_resume_text: str
    extracted_skills: List[SkillEntry]

class TargetState(TypedDict):
    raw_jd_text: str
    required_skills: List[str]  # ESCO URIs

class PipelineState(TypedDict):
    current: CurrentState
    target: TargetState
    skill_gap: List[str]                # ESCO URIs
    final_pathway: List[PathwayCourse]
    reasoning_trace: List[str]
    metrics: Metrics
