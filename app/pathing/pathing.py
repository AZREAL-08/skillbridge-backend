"""
Main pathing pipeline — orchestrates extraction, gap analysis, and Kahn sorting.
"""

from typing import List, Dict, Any, Optional
import networkx as nx
from app.state import PipelineState, CurrentState, TargetState, SkillEntry, PathwayCourse, Metrics, Question
from app.pathing.dag_builder import build_dag
from app.catalog.loader import load_catalog
from app.pathing.gap_analyzer import compute_skill_gap, get_active_subgraph, MASTERY_THRESHOLD
from app.pathing.kahn import kahn_priority_sort
from app.pathing.tracer import generate_reasoning_trace
from app.extractor.extractor import extract_skills, extract_skills_from_jd

def add_skipped_nodes(
    final_pathway: List[PathwayCourse], 
    current_skills: List[SkillEntry], 
    required_skills: List[str], 
    catalog: List[Dict[str, Any]]
) -> List[PathwayCourse]:
    """
    Identifies courses that were not in the active subgraph but are mastered
    by the user AND relevant to the JD. These should be shown as 'skipped'.
    """
    mastered = {s['taxonomy_id'] for s in current_skills if s['mastery_score'] >= MASTERY_THRESHOLD}
    target = set(required_skills)
    assigned_ids = {p['course_id'] for p in final_pathway}
    
    new_pathway = list(final_pathway)
    
    for course in catalog:
        cid = course['course_id']
        if cid in assigned_ids:
            continue
            
        # Course teaches something the JD needs AND user has mastered
        taught = set(course.get('skills_taught', []))
        if taught.intersection(mastered) and taught.intersection(target):
            # Calculate average mastery and confidence for this course
            course_masteries = []
            course_confidences = []
            for tid in taught:
                m = 0.0
                c = 0.7
                for s in current_skills:
                    if s['taxonomy_id'] == tid:
                        m = s['mastery_score']
                        c = s['confidence_score']
                        break
                course_masteries.append(m)
                course_confidences.append(c)
                
            avg_mastery = sum(course_masteries) / len(course_masteries) if course_masteries else 0.0
            avg_confidence = sum(course_confidences) / len(course_confidences) if course_confidences else 0.7
            
            new_pathway.append({
                "course_id": cid,
                "node_state": "skipped",
                "mastery_score": avg_mastery,
                "confidence_score": avg_confidence
            })
            
    return new_pathway

def run_pipeline(
    resume_text: str, 
    jd_text: str, 
    preferences: Optional[Dict[str, Any]] = None,
    domain_filter: Optional[str] = None,
    preference_questions: Optional[List[Question]] = None
) -> PipelineState:
    """
    Full pipeline: extract_skills → compute_gap → run_kahn → generate_traces
    """
    # 1. Load catalog and build DAG
    catalog = load_catalog()
    G = build_dag(catalog)
    
    # 2. Extract skills from Resume (Real Extraction)
    extracted_skills = extract_skills(resume_text, catalog)
    
    # 3. Extract skills from JD (Real Extraction)
    target_skill_entries = extract_skills_from_jd(jd_text)
    required_skills = [s['taxonomy_id'] for s in target_skill_entries if s.get('taxonomy_source') == 'emsi']
    
    # 4. Compute gap
    skill_gap = compute_skill_gap(extracted_skills, required_skills, catalog, domain_filter)
    
    # 5. Identify active subgraph
    node_states = get_active_subgraph(G, skill_gap, extracted_skills, domain_filter)
    active_node_ids = list(node_states.keys())
    
    # 6. Precompute metadata for Kahn sort
    course_metadata = {}
    gap_set = set(skill_gap)
    for cid in active_node_ids:
        meta = G.nodes[cid]
        
        taught = meta.get('skills_taught', [])
        gap_count = len([s for s in taught if s in gap_set])
        
        # Mastery of skills taught by this course
        masteries = []
        for tid in taught:
            m = 0.0
            for s in extracted_skills:
                if s['taxonomy_id'] == tid:
                    m = s['mastery_score']
                    break
            masteries.append(m)
        avg_mastery = sum(masteries) / len(masteries) if masteries else 0.0
        
        # Influence priority with preferences if needed
        # (For now, just pass through to metadata)
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
    total_hours = 0.0
    saved_hours = 0.0
    
    for cid in sorted_ids:
        state = node_states[cid]
        meta = G.nodes[cid]
        hours = meta.get('estimated_hours', 1.0)
        
        # Calculate mastery and confidence for this course in the final response
        taught = meta.get('skills_taught', [])
        
        course_masteries = []
        course_confidences = []
        for tid in taught:
            m = 0.0
            c = 0.5 # Default low confidence if not found
            for s in extracted_skills:
                if s['taxonomy_id'] == tid:
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
            total_hours += hours
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
        else:
            saved_hours += hours
            
    # 8.5 Add Skipped Nodes that weren't in the active subgraph
    final_pathway = add_skipped_nodes(final_pathway, extracted_skills, required_skills, catalog)
    
    # Recalculate saved_hours including independent skipped nodes
    mastered = {s['taxonomy_id'] for s in extracted_skills if s['mastery_score'] >= MASTERY_THRESHOLD}
    target = set(required_skills)
    for course in catalog:
        if course['course_id'] not in sorted_ids:
            taught = set(course.get('skills_taught', []))
            if taught.intersection(mastered) and taught.intersection(target):
                saved_hours += course.get('estimated_hours', 1.0)

    # 9. Compute Metrics
    baseline_courses = max(len(catalog), 30) # Always 30 (len of catalog)
    reduction_pct = 0.0
    if baseline_courses > 0:
        reduction_pct = ((baseline_courses - assigned_count) / baseline_courses) * 100
        
    metrics: Metrics = {
        "baseline_courses": baseline_courses,
        "assigned_courses": assigned_count,
        "reduction_pct": round(reduction_pct, 2),
        "total_hours": round(total_hours, 1),
        "saved_hours": round(saved_hours, 1)
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
        "metrics": metrics,
        "preference_questions": preference_questions or []
    }
    
    return state
