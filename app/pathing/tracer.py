"""
Reasoning trace string generator.
Produces human-readable explanations for each pathway decision.
"""

from typing import List, Dict, Any

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

def generate_reasoning_trace(
    course_id: str,
    node_state: str,
    course_metadata: Dict[str, Any],
    skill_gap: List[str],
    extracted_skills: List[Dict[str, Any]],
    dependent_course_title: str = ""
) -> str:
    """
    Generates human-readable reasoning traces for assigned and prerequisite nodes.
    :param course_id: The ID of the course.
    :param node_state: "assigned" or "prerequisite".
    :param course_metadata: Metadata for this course.
    :param skill_gap: User's skill gap taxonomy IDs.
    :param extracted_skills: User's extracted skills with mastery scores.
    :param dependent_course_title: Title of the course that depends on this one (for prerequisites).
    :return: Formatted trace string.
    """
    title = course_metadata.get('title', course_metadata.get('course_id', 'Unknown Course'))
    cid = course_id
    
    # Identify a relevant skill taught by this course
    taught = course_metadata.get('skills_taught', [])
    gap_set = set(skill_gap)
    relevant_skills = [s for s in taught if s in gap_set]
    
    if not relevant_skills:
        relevant_skills = taught[:1] # Fallback to first taught skill
        
    if not relevant_skills:
        # If the course metadata is truly minimal
        label = "core competencies"
        mastery = 0.0
    else:
        taxonomy_id = relevant_skills[0]
        # Find skill label and mastery from extracted_skills
        label = taxonomy_id
        mastery = 0.0
        for s in extracted_skills:
            if s['taxonomy_id'] == taxonomy_id:
                label = s.get('label', taxonomy_id)
                mastery = s.get('mastery_score', 0.0)
                break
                
    if node_state == 'assigned':
        return assigned_trace(title, cid, mastery, label)
    elif node_state == 'prerequisite':
        return prerequisite_trace(title, cid, mastery, label, dependent_course_title)
    
    return ""
