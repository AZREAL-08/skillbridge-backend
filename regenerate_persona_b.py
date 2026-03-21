#!/usr/bin/env python
"""
Regenerate Persona B extracted files (resume and JD).
"""

import json
from pathlib import Path
from dotenv import load_dotenv

from app.catalog.loader import load_catalog
from app.extractor.extractor import extract_skills, extract_skills_from_jd

load_dotenv()


def regenerate_persona_b_extractions():
    project_root = Path(__file__).resolve().parent
    
    # Regenerate resume
    print("=" * 80)
    print("REGENERATING PERSONA B RESUME EXTRACTION")
    print("=" * 80)
    
    resume_file = project_root / "data" / "persona_b_resume.txt"
    resume_output = project_root / "data" / "processed" / "persona_b_extracted_resume.json"
    
    print(f"\nLoading resume from {resume_file}")
    resume_text = resume_file.read_text(encoding="utf-8")
    
    catalog = load_catalog()
    print(f"Extracting skills from resume...")
    extracted_resume = extract_skills(resume_text, catalog)
    print(f"Extracted {len(extracted_resume)} resume skills")
    
    resume_output.write_text(json.dumps(extracted_resume, indent=2), encoding="utf-8")
    print(f"Saved to {resume_output}")
    
    # Show key skills
    power_bi = next((s for s in extracted_resume if "power bi" in s["label"].lower()), None)
    excel = next((s for s in extracted_resume if "excel" in s["label"].lower()), None)
    print(f"\nKey Skills:")
    if power_bi:
        print(f"  Power BI: {power_bi['mastery_score']:.2f} (target: <0.50)")
    if excel:
        print(f"  Excel: {excel['mastery_score']:.2f} (target: <0.50)")
    
    # Regenerate JD
    print("\n" + "=" * 80)
    print("REGENERATING PERSONA B JD EXTRACTION")
    print("=" * 80)
    
    jd_file = project_root / "data" / "persona_b_jd.txt"
    jd_output = project_root / "data" / "processed" / "persona_b_extracted_jd.json"
    
    print(f"\nLoading JD from {jd_file}")
    jd_text = jd_file.read_text(encoding="utf-8")
    
    print(f"Extracting skills from JD...")
    extracted_jd = extract_skills_from_jd(jd_text)
    print(f"Extracted {len(extracted_jd)} JD skills")
    
    jd_output.write_text(json.dumps(extracted_jd, indent=2), encoding="utf-8")
    print(f"Saved to {jd_output}")
    
    print("\nDone!")


if __name__ == "__main__":
    regenerate_persona_b_extractions()
