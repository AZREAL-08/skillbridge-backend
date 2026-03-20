from typing import List, Literal, TypedDict

class SkillEntry(TypedDict):
    esco_uri: str
    label: str
    mastery_score: float
    confidence_score: float

class PathwayCourse(TypedDict):
    course_id: str
    node_state: Literal["skipped", "assigned", "prerequisite"]
    mastery_score: float
    confidence_score: float

class Metrics(TypedDict):
    baseline_courses: int
    assigned_courses: int
    reduction_pct: float

class CurrentState(TypedDict):
    raw_resume_text: str
    extracted_skills: List[SkillEntry]

class TargetState(TypedDict):
    raw_jd_text: str
    required_skills: List[str]

class PipelineState(TypedDict):
    current: CurrentState
    target: TargetState
    skill_gap: List[str]
    final_pathway: List[PathwayCourse]
    reasoning_trace: List[str]
    metrics: Metrics
