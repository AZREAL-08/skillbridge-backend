"""
Standalone JD extraction test.

Tests that extract_skills_from_jd():
1. Extracts skills from JD text
2. Does NOT run mastery scoring (all mastery_score = 0.0)
3. Saves to data/processed/

Run: python -m tests.test_jd_extraction
"""

import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from app.extractor.extractor import extract_skills_from_jd
from app.pathing.gap_analyzer import NOISE_TAXONOMY_IDS


def test_persona_a_jd():
    """Persona A JD: Full-Stack Software Engineer."""
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    jd_text = (project_root / "data" / "persona_a_jd.txt").read_text(encoding="utf-8")

    extracted = extract_skills_from_jd(jd_text)
    assert extracted, "JD extractor should return at least one skill"

    # Save intermediate result
    out_file = output_dir / "persona_a_extracted_jd.json"
    out_file.write_text(json.dumps(extracted, indent=2), encoding="utf-8")
    print(f"\n✅ Saved {len(extracted)} JD skills to {out_file}")

    print("\n── Persona A JD Skills ──")
    for skill in extracted:
        flag = " ⚠️ NOISE" if skill["taxonomy_id"] in NOISE_TAXONOMY_IDS else ""
        print(f"  {skill['label']:30s} → mastery={skill['mastery_score']:.2f}  id={skill['taxonomy_id']}{flag}")

    # Assert: ALL mastery scores should be 0.0 (JD has no mastery)
    for skill in extracted:
        assert skill["mastery_score"] == 0.0, (
            f"JD skill '{skill['label']}' has mastery {skill['mastery_score']} — should be 0.0"
        )
    print("\n✓ All JD mastery scores are 0.0 (correct)")

    # Check which of the extracted skills are noise
    noise_in_jd = [s for s in extracted if s["taxonomy_id"] in NOISE_TAXONOMY_IDS]
    clean = [s for s in extracted if s["taxonomy_id"] not in NOISE_TAXONOMY_IDS]
    
    print(f"\n  Total JD skills extracted: {len(extracted)}")
    print(f"  Noise skills (will be filtered in gap analysis): {len(noise_in_jd)}")
    print(f"  Clean skills (will drive routing): {len(clean)}")

    # These junk words should be flagged as noise
    noise_labels = {s["label"].lower() for s in noise_in_jd}
    for expected_noise in ["write", "nice", "scalable", "manage", "software", "backend"]:
        if expected_noise in noise_labels:
            print(f"  ✓ '{expected_noise}' correctly identified as noise")

    print("\n✅ Persona A JD extraction assertions passed!")
    return extracted


def test_persona_b_jd():
    """Persona B JD: Supply Chain Operations Manager."""
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    jd_text = (project_root / "data" / "persona_b_jd.txt").read_text(encoding="utf-8")

    extracted = extract_skills_from_jd(jd_text)
    assert extracted, "JD extractor should return at least one skill"

    # Save intermediate result
    out_file = output_dir / "persona_b_extracted_jd.json"
    out_file.write_text(json.dumps(extracted, indent=2), encoding="utf-8")
    print(f"\n✅ Saved {len(extracted)} JD skills to {out_file}")

    print("\n── Persona B JD Skills ──")
    for skill in extracted:
        flag = " ⚠️ NOISE" if skill["taxonomy_id"] in NOISE_TAXONOMY_IDS else ""
        print(f"  {skill['label']:30s} → mastery={skill['mastery_score']:.2f}  id={skill['taxonomy_id']}{flag}")

    # Assert: ALL mastery scores should be 0.0
    for skill in extracted:
        assert skill["mastery_score"] == 0.0, (
            f"JD skill '{skill['label']}' has mastery {skill['mastery_score']} — should be 0.0"
        )
    print("\n✓ All JD mastery scores are 0.0 (correct)")

    print("\n✅ Persona B JD extraction assertions passed!")
    return extracted


def main():
    print("=" * 60)
    print("JD EXTRACTION TEST — No Mastery Scoring")
    print("=" * 60)

    print("\n── Testing Persona A JD ──")
    test_persona_a_jd()

    print("\n" + "=" * 60)
    print("\n── Testing Persona B JD ──")
    test_persona_b_jd()


if __name__ == "__main__":
    main()
