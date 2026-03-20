from typing import List, Set, Dict, Any
import networkx as nx
from app.state import SkillEntry

MASTERY_THRESHOLD = 0.85

def get_skill_gap(extracted_skills: List[SkillEntry], required_skills: List[str]) -> List[str]:
    """
    Computes skill gap: required_skills - mastered_skills.
    :param extracted_skills: Skills from user's resume with mastery scores.
    :param required_skills: Skills required by the JD (ESCO URIs).
    :return: List of ESCO URIs in the gap.
    """
    mastered_skills = {
        s['esco_uri'] for s in extracted_skills if s['mastery_score'] >= MASTERY_THRESHOLD
    }
    gap = [s for s in required_skills if s not in mastered_skills]
    return gap

def get_active_subgraph(
    G: nx.DiGraph, 
    skill_gap: List[str], 
    extracted_skills: List[SkillEntry]
) -> Dict[str, str]:
    """
    Identifies assigned, prerequisite, and skipped courses.
    :param G: The full catalog DAG.
    :param skill_gap: List of ESCO URIs in the user's gap.
    :param extracted_skills: Full list of user's extracted skills.
    :return: Dictionary mapping course_id to node_state ("assigned", "prerequisite", "skipped").
    """
    gap_set = set(skill_gap)
    mastered_skills = {
        s['esco_uri'] for s in extracted_skills if s['mastery_score'] >= MASTERY_THRESHOLD
    }
    
    # Direct candidates: courses that teach at least one skill in the gap
    assigned_ids = set()
    for cid, data in G.nodes(data=True):
        taught = set(data.get('skills_taught', []))
        if taught.intersection(gap_set):
            assigned_ids.add(cid)
            
    active_nodes = {} # course_id -> state

    def add_recursive(cid, is_assigned):
        # Determine mastery status for this course
        taught = set(G.nodes[cid].get('skills_taught', []))
        is_fully_mastered = len(taught) > 0 and taught.issubset(mastered_skills)
        
        target_state = 'assigned' if is_assigned else 'prerequisite'
        if is_fully_mastered:
            target_state = 'skipped'
            
        # Merge with existing state if already visited
        if cid in active_nodes:
            old_state = active_nodes[cid]
            if old_state == 'skipped':
                # If it was skipped, it stays skipped (mastery is absolute)
                return
            if target_state == 'assigned':
                active_nodes[cid] = 'assigned'
            # If target is prerequisite and old is assigned, it stays assigned
            # If target is skipped, it becomes skipped
            if target_state == 'skipped':
                active_nodes[cid] = 'skipped'
        else:
            active_nodes[cid] = target_state
            
        # If not skipped, we must ensure its prerequisites are considered
        if active_nodes[cid] != 'skipped':
            for prereq in G.predecessors(cid):
                add_recursive(prereq, False)

    for cid in assigned_ids:
        add_recursive(cid, True)
        
    return active_nodes
