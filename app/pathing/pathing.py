"""
Main pathing pipeline — orchestrates extraction, gap analysis, and Kahn sorting.
"""

from typing import List, Dict, Any
import networkx as nx
from app.state import PipelineState, CurrentState, TargetState, SkillEntry, PathwayCourse, Metrics
from app.pathing.dag_builder import build_dag
from app.catalog.loader import load_catalog
from app.pathing.gap_analyzer import compute_skill_gap, get_active_subgraph
from app.pathing.kahn import kahn_priority_sort
from app.pathing.tracer import generate_reasoning_trace

def mock_extract_skills(resume_text: str) -> List[SkillEntry]:
    """
    Mock extraction for development until Person 2 is ready.
    Provides deterministic data for testing the pathing engine.
    """
    return [
        {"esco_uri": "python-001", "label": "Python Programming", "mastery_score": 0.4, "confidence_score": 0.9},
        {"esco_uri": "git-001", "label": "Git Version Control", "mastery_score": 0.9, "confidence_score": 0.95},
        {"esco_uri": "sql-001", "label": "SQL Database", "mastery_score": 0.2, "confidence_score": 0.8},
        {"esco_uri": "logic-001", "label": "Logic", "mastery_score": 0.95, "confidence_score": 0.99},
        {"esco_uri": "db-001", "label": "Database Knowledge", "mastery_score": 0.1, "confidence_score": 0.7}
    ]

def run_pipeline(resume_text: str, jd_text: str) -> PipelineState:
    """
    Full pipeline: extract_skills → compute_gap → run_kahn → generate_traces
    """
    # 1. Load catalog and build DAG
    catalog = load_catalog()
    # For dev, if catalog is empty, use a tiny mock catalog
    if not catalog:
        catalog = [
            {
                "course_id": "C-PY-101", "title": "Intro to Python", 
                "skills_taught": ["python-001"], "prerequisites": ["C-PY-000"], 
                "estimated_hours": 10.0, "difficulty": "Beginner", "domain": "Tech"
            },
            {
                "course_id": "C-PY-000", "title": "Programming Basics", 
                "skills_taught": ["logic-001"], "prerequisites": [], 
                "estimated_hours": 5.0, "difficulty": "Beginner", "domain": "Tech"
            },
            {
                "course_id": "C-SQL-101", "title": "SQL Basics", 
                "skills_taught": ["sql-001"], "prerequisites": ["C-DB-000"], 
                "estimated_hours": 5.0, "difficulty": "Beginner", "domain": "Data"
            },
            {
                "course_id": "C-DB-000", "title": "Database Fundamentals", 
                "skills_taught": ["db-001"], "prerequisites": [], 
                "estimated_hours": 2.0, "difficulty": "Beginner", "domain": "Data"
            },
            {
                "course_id": "C-GIT-101", "title": "Git for Pros", 
                "skills_taught": ["git-001"], "prerequisites": [], 
                "estimated_hours": 2.0, "difficulty": "Beginner", "domain": "Tech"
            }
        ]
    G = build_dag(catalog)
    
    # 2. Extract skills (Mocked for now)
    extracted_skills = mock_extract_skills(resume_text)
    
    # 3. Target skills (Mocked: JD requires Python, SQL, and Git)
    # In real implementation, this would come from an extractor for JD text.
    required_skills = ["python-001", "sql-001", "git-001"]
    
    # 4. Compute gap
    skill_gap = compute_skill_gap(extracted_skills, required_skills)
    
    # 5. Identify active subgraph
    node_states = get_active_subgraph(G, skill_gap, extracted_skills)
    active_node_ids = list(node_states.keys())
    
    # 6. Precompute metadata for Kahn sort
    course_metadata = {}
    gap_set = set(skill_gap)
    for cid in active_node_ids:
        meta = G.nodes[cid]
        
        taught = meta.get('skills_taught', [])
        gap_count = len([s for s in taught if s in gap_set])
        
        # Mastery of skills taught by this course
        # If multiple skills, we take average for priority purposes
        masteries = []
        for esco in taught:
            m = 0.0
            for s in extracted_skills:
                if s['esco_uri'] == esco:
                    m = s['mastery_score']
                    break
            masteries.append(m)
        avg_mastery = sum(masteries) / len(masteries) if masteries else 0.0
        
        course_metadata[cid] = {
            "prerequisites": meta.get('prerequisites', []),
            "gap_count": gap_count,
            "mastery": avg_mastery,
            "hours": meta.get('estimated_hours', 1.0)
        }
        
    # 7. Topological Sort
    sorted_ids = kahn_priority_sort(active_node_ids, course_metadata)
    
    # 8. Build final pathway
    final_pathway: List[PathwayCourse] = []
    reasoning_traces: List[str] = []
    assigned_count = 0
    
    for cid in sorted_ids:
        state = node_states[cid]
        meta = G.nodes[cid]
        
        # Calculate mastery and confidence for this course in the final response
        taught = meta.get('skills_taught', [])
        
        course_masteries = []
        course_confidences = []
        for esco in taught:
            m = 0.0
            c = 0.5 # Default low confidence if not found
            for s in extracted_skills:
                if s['esco_uri'] == esco:
                    m = s['mastery_score']
                    c = s['confidence_score']
                    break
            course_masteries.append(m)
            course_confidences.append(c)
            
        avg_mastery = sum(course_masteries) / len(course_masteries) if course_masteries else 0.0
        avg_confidence = sum(course_confidences) / len(course_confidences) if course_confidences else 0.7
        
        final_pathway.append({
            "course_id": cid,
            "node_state": state, # type: ignore
            "mastery_score": avg_mastery,
            "confidence_score": avg_confidence
        })
        
        if state != 'skipped':
            assigned_count += 1
            # Find a dependent course title for prerequisites
            dep_title = ""
            if state == 'prerequisite':
                for succ in G.successors(cid):
                    if succ in sorted_ids and node_states[succ] != 'skipped':
                        dep_title = G.nodes[succ].get('title', succ)
                        break
            
            trace = generate_reasoning_trace(
                cid, state, meta, skill_gap, extracted_skills, dep_title
            )
            reasoning_traces.append(trace)
            
    # 9. Compute Metrics
    baseline_courses = max(len(catalog), 30) # PDF says "Always 30 (len of catalog)"
    reduction_pct = 0.0
    if baseline_courses > 0:
        reduction_pct = ((baseline_courses - assigned_count) / baseline_courses) * 100
        
    metrics: Metrics = {
        "baseline_courses": baseline_courses,
        "assigned_courses": assigned_count,
        "reduction_pct": round(reduction_pct, 2)
    }
    
    # 10. Assemble State
    state: PipelineState = {
        "current": {
            "raw_resume_text": resume_text,
            "extracted_skills": extracted_skills
        },
        "target": {
            "raw_jd_text": jd_text,
            "required_skills": required_skills
        },
        "skill_gap": skill_gap,
        "final_pathway": final_pathway,
        "reasoning_trace": reasoning_traces,
        "metrics": metrics
    }
    
    return state
