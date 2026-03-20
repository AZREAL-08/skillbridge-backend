"""
Reasoning trace string generator.
Produces human-readable explanations for each pathway decision.
"""


def assigned_trace(
    course_title: str,
    course_id: str,
    mastery_score: float,
    skill_label: str,
) -> str:
    """Template for directly-assigned courses."""
    return (
        f"{course_title} ({course_id}) added: User has {mastery_score:.2f} "
        f"mastery of {skill_label}. This skill is directly required by the target JD."
    )


def prerequisite_trace(
    course_title: str,
    course_id: str,
    mastery_score: float,
    skill_label: str,
    dependent_course_title: str,
) -> str:
    """Template for prerequisite pull-in courses."""
    return (
        f"{course_title} ({course_id}) added: User has {mastery_score:.2f} "
        f"mastery of {skill_label}. Required as a prerequisite for "
        f"{dependent_course_title}, which addresses a gap in the target JD."
    )
