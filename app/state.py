"""
ALL TypedDicts live here — shared contract between backend modules.
Frontend mirrors these in lib/types.ts.
"""

from typing import TypedDict, List, Literal, Dict, Any, Optional


class SkillEntry(TypedDict):
    taxonomy_id: str  # EMSI ID
    taxonomy_source: Literal["emsi", "inferred"]
    label: str
    mastery_score: float  # 0.0–1.0
    confidence_score: float  # 0.0–1.0


class PathwayCourse(TypedDict):
    course_id: str
    node_state: Literal["skipped", "assigned", "prerequisite"]
    mastery_score: float
    confidence_score: float


class Metrics(TypedDict):
    baseline_courses: int
    assigned_courses: int
    reduction_pct: float
    total_hours: float  # NEW in v2.0
    saved_hours: float  # NEW in v2.0


class CurrentState(TypedDict):
    raw_resume_text: str
    extracted_skills: List[SkillEntry]


class TargetState(TypedDict):
    raw_jd_text: str
    required_skills: List[str]


class Question(TypedDict):  # NEW in v2.0
    id: str
    text: str
    options: List[str]


class PipelineState(TypedDict):
    current: CurrentState
    target: TargetState
    skill_gap: List[str]
    final_pathway: List[PathwayCourse]
    reasoning_trace: List[str]
    metrics: Metrics
    preference_questions: List[Question]  # NEW in v2.0


class StoredJD(TypedDict):  # NEW in v2.0 (HR Flow)
    jd_id: str
    role_title: str
    company: str
    department: str
    domain: Literal["technical", "operations"]
    raw_text: str
    required_skills: List[SkillEntry]
    created_at: str
    is_deleted: Optional[bool]  # For soft-delete support

