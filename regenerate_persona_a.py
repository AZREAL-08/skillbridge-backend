#!/usr/bin/env python
"""
Regenerate Persona A extracted files to verify all fixes.
"""

import json
from pathlib import Path
from dotenv import load_dotenv

from app.catalog.loader import load_catalog
from app.extractor.extractor import extract_skills, extract_skills_from_jd
from app.pathing.gap_analyzer import compute_skill_gap, get_active_subgraph, NOISE_TAXONOMY_IDS

load_dotenv()


def regenerate_persona_a():
    project_root = Path(__file__).resolve().parent
    
    # Regenerate resume
    print("=" * 80)
    print("PERSONA A - REGENERATING EXTRACTIONS")
    print("=" * 80)
    
    resume_file = project_root / "data" / "persona_a_resume.txt"
    resume_output = project_root / "data" / "processed" / "persona_a_extracted_resume.json"
    jd_file = project_root / "data" / "persona_a_jd.txt"
    jd_output = project_root / "data" / "processed" / "persona_a_extracted_jd.json"
    
    print(f"\nLoading Persona A resume...")
    resume_text = resume_file.read_text(encoding="utf-8")
    
    catalog = load_catalog()
    print(f"Extracting resume skills...")
    extracted_resume = extract_skills(resume_text, catalog)
    print(f"Extracted {len(extracted_resume)} resume skills")
    resume_output.write_text(json.dumps(extracted_resume, indent=2), encoding="utf-8")
    
    print(f"\nLoading Persona A JD...")
    jd_text = jd_file.read_text(encoding="utf-8")
    print(f"Extracting JD skills...")
    extracted_jd = extract_skills_from_jd(jd_text)
    print(f"Extracted {len(extracted_jd)} JD skills")
    jd_output.write_text(json.dumps(extracted_jd, indent=2), encoding="utf-8")
    
    # Compute gap
    jd_required = [s["taxonomy_id"] for s in extracted_jd if s.get("taxonomy_source") == "emsi"]
    skill_gap = compute_skill_gap(extracted_resume, jd_required)
    required_filtered = [s for s in jd_required if s not in NOISE_TAXONOMY_IDS]
    
    print("\n" + "=" * 80)
    print("PERSONA A ANALYSIS")
    print("=" * 80)
    
    print(f"\nJD: {len(jd_required)} required → {len(required_filtered)} after noise")
    print(f"Gap: {len(skill_gap)} missing skills")
    
    # Check key skills
    print(f"\nKey Resume Skills:")
    for skill in sorted(extracted_resume, key=lambda s: s["mastery_score"], reverse=True)[:15]:
        label = skill["label"]
        mastery = skill["mastery_score"]
        
        # Check for specific ones the user mentioned
        status = ""
        if "node" in label.lower() or "js" in label.lower():
            status = f" ← Node.js (target <0.50)" if mastery > 0.50 else f" ← Node.js OK"
        elif "react" in label.lower():
            status = f" ← React (target >=0.85)" if mastery < 0.85 else f" ← React GOOD"
        elif "git" in label.lower():
            status = f" ← Git (target ~0.95)" if mastery < 0.90 else f" ← Git GOOD"
        elif "linux" in label.lower():
            status = f" ← Linux (target ~0.95)" if mastery < 0.90 else f" ← Linux GOOD"
        
        print(f"  {label}: {mastery:.2f}{status}")
    
    print("\nDone!")


if __name__ == "__main__":
    regenerate_persona_a()
