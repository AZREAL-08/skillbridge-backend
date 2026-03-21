#!/usr/bin/env python
"""
Regenerate just the Persona B resume extraction to test negation detection.
"""

import json
from pathlib import Path
from dotenv import load_dotenv

from app.catalog.loader import load_catalog
from app.extractor.extractor import extract_skills

load_dotenv()


def regenerate_persona_b_resume():
    project_root = Path(__file__).resolve().parent
    resume_file = project_root / "data" / "persona_b_resume.txt"
    output_file = project_root / "data" / "processed" / "persona_b_extracted_resume.json"
    
    print("=" * 80)
    print("REGENERATING PERSONA B RESUME EXTRACTION")
    print("=" * 80)
    
    # Load resume text
    print(f"\nLoading resume from {resume_file}")
    resume_text = resume_file.read_text(encoding="utf-8")
    print(f"Resume length: {len(resume_text)} chars")
    
    # Load catalog for bloom level mapping
    print("\nLoading catalog for bloom level mapping...")
    catalog = load_catalog()
    print(f"Catalog loaded: {len(catalog)} courses")
    
    # Extract skills from resume
    print("\nExtracting skills from resume (may take 1-2 minutes due to Groq call)...")
    extracted_skills = extract_skills(resume_text, catalog)
    print(f"Extracted {len(extracted_skills)} skills")
    
    # Save to file
    print(f"\nSaving to {output_file}")
    output_file.write_text(json.dumps(extracted_skills, indent=2), encoding="utf-8")
    
    # Print summary with critical skills
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal extracted skills: {len(extracted_skills)}")
    
    # Find and print Power BI and Excel
    critical_skills = {}
    for skill in extracted_skills:
        label = skill["label"].lower()
        if "power" in label or "excel" in label or "pivot" in label or "data" in label and "analysis" in label:
            critical_skills[label] = skill["mastery_score"]
    
    print(f"\nCritical Skills (should be low due to 'no experience' text):")
    for label in sorted(critical_skills.keys()):
        mastery = critical_skills[label]
        status = "✓ LOW" if mastery < 0.50 else "✗ TOO HIGH"
        print(f"  {status}: {label} = {mastery:.2f}")
    
    # Show all skills sorted by mastery
    print(f"\nAll Skills (sorted by mastery score):")
    for skill in sorted(extracted_skills, key=lambda s: s["mastery_score"], reverse=True):
        print(f"  {skill['label']}: {skill['mastery_score']:.2f}")
    
    print("\nDone!")


if __name__ == "__main__":
    regenerate_persona_b_resume()
