#!/usr/bin/env python
"""
Run full discovery pipeline for Persona B and generate:
- persona_b_diff_output.json
- persona_b_kahn.json
"""

import json
from pathlib import Path
from dotenv import load_dotenv

from app.catalog.loader import load_catalog
from app.pathing.dag_builder import build_dag
from app.pathing.gap_analyzer import compute_skill_gap, get_active_subgraph, NOISE_TAXONOMY_IDS
from app.pathing.kahn import kahn_priority_sort

load_dotenv()


def run_persona_b_pipeline():
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "data" / "processed"
    
    # Load existing extractions
    resume_file = output_dir / "persona_b_extracted_resume.json"
    jd_file = output_dir / "persona_b_extracted_jd.json"
    
    if not resume_file.exists() or not jd_file.exists():
        print(f"ERROR: Extracted files not found!")
        print(f"  Resume: {resume_file} (exists={resume_file.exists()})")
        print(f"  JD: {jd_file} (exists={jd_file.exists()})")
        return
    
    print(f"Loading extracted resume from {resume_file}")
    extracted_resume = json.loads(resume_file.read_text(encoding="utf-8"))
    print(f"  Loaded {len(extracted_resume)} resume skills")
    
    print(f"Loading extracted JD from {jd_file}")
    extracted_jd = json.loads(jd_file.read_text(encoding="utf-8"))
    print(f"  Loaded {len(extracted_jd)} JD skills")
    
    # Compute skill gap
    print("\nComputing skill gap...")
    jd_required = [s["taxonomy_id"] for s in extracted_jd if s.get("taxonomy_source") == "emsi"]
    print(f"  JD required skills (EMSI): {len(jd_required)}")
    
    required_filtered = [s for s in jd_required if s not in NOISE_TAXONOMY_IDS]
    print(f"  After noise filtering: {len(required_filtered)}")
    
    skill_gap = compute_skill_gap(extracted_resume, jd_required)
    print(f"  Skill gap: {len(skill_gap)}")
    
    # Write diff output
    print("\nGenerating diff output...")
    diff_payload = {
        "required_skills": jd_required,
        "required_skills_after_noise_filter": required_filtered,
        "skill_gap": skill_gap,
        "counts": {
            "required_raw": len(jd_required),
            "required_filtered": len(required_filtered),
            "gap": len(skill_gap),
        },
    }
    diff_file = output_dir / "persona_b_diff_output.json"
    diff_file.write_text(json.dumps(diff_payload, indent=2), encoding="utf-8")
    print(f"  Saved: {diff_file}")
    
    # Load catalog and build DAG
    print("\nBuilding course DAG...")
    catalog = load_catalog()
    G = build_dag(catalog)
    print(f"  DAG has {len(G)} courses")
    
    # Compute active subgraph (assigned/prerequisite/skipped courses)
    print("\nComputing active subgraph...")
    node_states = get_active_subgraph(G, skill_gap, extracted_resume)
    active_node_ids = list(node_states.keys())
    print(f"  Active nodes: {len(active_node_ids)}")
    print(f"    Assigned: {sum(1 for s in node_states.values() if s == 'assigned')}")
    print(f"    Prerequisite: {sum(1 for s in node_states.values() if s == 'prerequisite')}")
    print(f"    Skipped: {sum(1 for s in node_states.values() if s == 'skipped')}")
    
    # Compute course metadata for Kahn
    print("\nComputing course metadata for Kahn...")
    gap_set = set(skill_gap)
    jd_set = {s["taxonomy_id"] for s in extracted_jd if s.get("taxonomy_source") == "emsi"}
    
    course_metadata = {}
    for cid in active_node_ids:
        meta = G.nodes[cid]
        taught = meta.get("skills_taught", [])
        gap_count = len([s for s in taught if s in gap_set and s in jd_set])
        
        masteries = []
        for skill_id in taught:
            mastery = 0.0
            for skill in extracted_resume:
                if skill["taxonomy_id"] == skill_id:
                    mastery = skill["mastery_score"]
                    break
            masteries.append(mastery)
        
        avg_mastery = sum(masteries) / len(masteries) if masteries else 0.0
        
        course_metadata[cid] = {
            "prerequisites": meta.get("prerequisites", []),
            "gap_count": gap_count,
            "mastery": avg_mastery,
            "hours": meta.get("estimated_hours", 1.0),
            "bloom_level": meta.get("bloom_level", 3),
        }
    
    # Run Kahn sort
    print("\nRunning Kahn priority sort...")
    kahn_order = kahn_priority_sort(active_node_ids, course_metadata) if active_node_ids else []
    print(f"  Kahn order: {len(kahn_order)} courses")
    
    # Write Kahn output
    print("\nGenerating Kahn output...")
    kahn_payload = {
        "input_diff_file": str(diff_file),
        "active_node_ids": active_node_ids,
        "node_states": node_states,
        "kahn_order": kahn_order,
    }
    kahn_file = output_dir / "persona_b_kahn.json"
    kahn_file.write_text(json.dumps(kahn_payload, indent=2), encoding="utf-8")
    print(f"  Saved: {kahn_file}")
    
    # Print summary with key courses
    print("\n" + "="*80)
    print("PERSONA B PIPELINE SUMMARY")
    print("="*80)
    print(f"\nJD Requirements: {len(jd_required)} skills → {len(required_filtered)} after noise")
    print(f"Resume Skills: {len(extracted_resume)}")
    print(f"Skill Gap: {len(skill_gap)} missing skills")
    print(f"\nCourse Pathway:")
    print(f"  Assigned: {sum(1 for s in node_states.values() if s == 'assigned')} courses")
    print(f"  Prerequisites: {sum(1 for s in node_states.values() if s == 'prerequisite')} courses")
    print(f"  Skipped: {sum(1 for s in node_states.values() if s == 'skipped')} courses")
    
    # Show top skills in resume
    resume_by_mastery = sorted(extracted_resume, key=lambda s: s["mastery_score"], reverse=True)
    print(f"\nTop 10 Resume Skills by Mastery:")
    for skill in resume_by_mastery[:10]:
        print(f"  {skill['label']}: {skill['mastery_score']:.2f}")
    
    # Show Kahn order (top 10)
    print(f"\nTop 10 Recommended Courses (Kahn Order):")
    for i, cid in enumerate(kahn_order[:10], 1):
        state = node_states.get(cid, "unknown")
        meta = G.nodes[cid]
        name = meta.get("course_name", cid)
        gap = course_metadata[cid]["gap_count"]
        print(f"  {i}. {name} ({state}, {gap} gap skills)")
    
    print("\nDone!")


if __name__ == "__main__":
    run_persona_b_pipeline()
