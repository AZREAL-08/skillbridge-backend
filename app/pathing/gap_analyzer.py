"""
Set intersection logic for skill gap analysis.
Skill Gap IDs = Target JD IDs − User Mastered IDs
Only EMSI-mapped skills participate in set intersection.
Recursive subgraph pull-ins for prerequisites.

Includes noise filtering to remove generic non-skill EMSI IDs from
JD required skills before gap analysis.
"""

from typing import List, Set, Dict, Any
import logging
import networkx as nx
from app.state import SkillEntry

logger = logging.getLogger(__name__)

# Mastery threshold: >= 0.85 means "mastered" (bypassed)
MASTERY_THRESHOLD = 0.85

# ── Noise Filtering ──────────────────────────────────────────────────────────
# These EMSI IDs are generic English words that SkillNER over-extracts from
# JD text. They are NOT real skills and should never drive gap analysis or
# course routing. This list was curated from observed false positives.

NOISE_TAXONOMY_IDS: Set[str] = {
    "KS4425C7820LCHZS7VGX",  # "write"
    "KS4400B70JFSWXTYH0P2",  # "nice"
    "KS124RX787SQ1WVD8XF6",  # "scalable"
    "KS124ZQ6RS1V188JSCTX",  # "best practices"
    "KS1218W78FGVPVP2KXPX",  # "manage"
    "KS124T66K0SY8B8G77LX",  # "high performance"
    "KS7R8G2D52QH187SED9R",  # "backend"
    "KS440QS66YCBN23Y8K25",  # "software"
    "ES76F4C57A88877D6D64",  # "professional"
    "KS4424T6KPTTQ1NKM0XK",  # "workflow"
    "KS125R96VYB8GM02DSMV",  # "layers"
    "KS440H66BML35BBRFCTK",  # "server side"
    "KS1227V6GK3GDKLR52KN",  # "rendering"
    "KS122086PPY11B2M1G6N",  # "libraries"
    "KS8TU0J0IOFIJUU9VS4H",  # "load time"
    "KS440FZ66QFPWRRTYF6J",  # "semantic"
    "KS1256Z6HDJ91HWJ615M",  # "product teams"
    "KS125716TLTGH6SDHJD1",  # "integration"
    "KS441ZY6P0PDB5DWTRB8",  # "web application"
    "ESC67B4D284220100378",  # "web app"
    "KS2UJ31ABTM22Y2MHEQM",  # "custom component"
    "KS1228S6YKWXMH4M4DL7",  # "conflict resolution"
}


def filter_noise_skills(skill_ids: List[str]) -> List[str]:
    """
    Remove known junk EMSI IDs from a list of skill taxonomy IDs.
    Returns filtered list with noise removed. Logs count of filtered items.
    """
    filtered = [sid for sid in skill_ids if sid not in NOISE_TAXONOMY_IDS]
    removed = len(skill_ids) - len(filtered)
    if removed > 0:
        logger.info(
            "[GapAnalyzer] Filtered %d noise skills from %d total (kept %d).",
            removed, len(skill_ids), len(filtered),
        )
    return filtered


def compute_skill_gap(extracted_skills: List[SkillEntry], required_skills: List[str]) -> List[str]:
    """
    Computes skill gap: required_skills - mastered_skills.
    Only EMSI-mapped skills (taxonomy_source == "emsi") participate in the
    set intersection. Inferred skills do not drive gap analysis.

    Noise skills are filtered from required_skills before computing the gap.

    :param extracted_skills: Skills from user's resume with mastery scores.
    :param required_skills: Skills required by the JD (EMSI taxonomy IDs).
    :return: List of EMSI taxonomy IDs in the gap.
    """
    # Filter noise from JD required skills
    clean_required = filter_noise_skills(required_skills)

    mastered_skills = {
        s['taxonomy_id'] for s in extracted_skills
        if s.get('taxonomy_source') == 'emsi' and s['mastery_score'] >= MASTERY_THRESHOLD
    }
    gap = [s for s in clean_required if s not in mastered_skills]
    return gap

def get_active_subgraph(
    G: nx.DiGraph, 
    skill_gap: List[str], 
    extracted_skills: List[SkillEntry]
) -> Dict[str, str]:
    """
    Identifies assigned, prerequisite, and skipped courses.
    :param G: The full catalog DAG.
    :param skill_gap: List of EMSI taxonomy IDs in the user's gap.
    :param extracted_skills: Full list of user's extracted skills.
    :return: Dictionary mapping course_id to node_state ("assigned", "prerequisite", "skipped").
    """
    gap_set = set(skill_gap)
    mastered_skills = {
        s['taxonomy_id'] for s in extracted_skills
        if s.get('taxonomy_source') == 'emsi' and s['mastery_score'] >= MASTERY_THRESHOLD
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
