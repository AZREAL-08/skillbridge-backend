"""
Standalone mastery scoring test.

Tests the profile-classification + bloom-decay mastery architecture:
1. Loads persona resume text
2. Runs extract_skills() with catalog
3. Saves to data/processed/
4. Asserts key mastery thresholds

Run: python -m tests.test_mastery
"""

import json
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from app.extractor.extractor import extract_skills
from app.catalog.loader import load_catalog


def test_persona_a_mastery():
    """Persona A: Senior Frontend Engineer — base should be senior_professional (~0.85)."""
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_text = (project_root / "data" / "persona_a_resume.txt").read_text(encoding="utf-8")
    catalog = load_catalog()
    assert catalog, "Catalog must be available at data/catalog.json"

    extracted = extract_skills(resume_text, catalog)
    assert extracted, "Extractor should return at least one skill"

    # Save intermediate result
    out_file = output_dir / "persona_a_extracted_resume.json"
    out_file.write_text(json.dumps(extracted, indent=2), encoding="utf-8")
    print(f"\n✅ Saved {len(extracted)} skills to {out_file}")

    # Build lookup by label
    scores = {s["label"].lower(): s["mastery_score"] for s in extracted}

    print("\n── Persona A Mastery Scores ──")
    for label, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {label:30s} → {score:.2f}")

    # Key assertions for a senior_professional profile
    # Git: daily workflow tool, bloom 1 → should be very high
    if "git" in scores:
        assert scores["git"] >= 0.75, f"Git should be ≥ 0.75, got {scores['git']}"
        print(f"\n✓ git = {scores['git']:.2f} (expected ≥ 0.75)")

    # React: core skill, bloom 3 → moderate-high
    if "react js" in scores:
        assert scores["react js"] >= 0.60, f"React should be ≥ 0.60, got {scores['react js']}"
        print(f"✓ react js = {scores['react js']:.2f} (expected ≥ 0.60)")

    # JavaScript: core skill, bloom 2 → high
    if "javascript" in scores:
        assert scores["javascript"] >= 0.70, f"JavaScript should be ≥ 0.70, got {scores['javascript']}"
        print(f"✓ javascript = {scores['javascript']:.2f} (expected ≥ 0.70)")

    # REST APIs: consumed dozens, bloom 2 → should be moderate+
    if "rest apis" in scores:
        assert scores["rest apis"] >= 0.60, f"REST APIs should be ≥ 0.60, got {scores['rest apis']}"
        print(f"✓ rest apis = {scores['rest apis']:.2f} (expected ≥ 0.60)")

    # GitHub: CI/CD integration, explicit mention
    if "github" in scores:
        assert scores["github"] >= 0.60, f"GitHub should be ≥ 0.60, got {scores['github']}"
        print(f"✓ github = {scores['github']:.2f} (expected ≥ 0.60)")

    print("\n✅ Persona A mastery assertions passed!")
    return extracted


def test_persona_b_mastery():
    """Persona B: Warehouse Operations Supervisor — base should be senior_professional or mid_professional."""
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_text = (project_root / "data" / "persona_b_resume.txt").read_text(encoding="utf-8")
    catalog = load_catalog()
    assert catalog, "Catalog must be available at data/catalog.json"

    extracted = extract_skills(resume_text, catalog)
    assert extracted, "Extractor should return at least one skill"

    # Save intermediate result
    out_file = output_dir / "persona_b_extracted_resume.json"
    out_file.write_text(json.dumps(extracted, indent=2), encoding="utf-8")
    print(f"\n✅ Saved {len(extracted)} skills to {out_file}")

    # Build lookup by label
    scores = {s["label"].lower(): s["mastery_score"] for s in extracted}

    print("\n── Persona B Mastery Scores ──")
    for label, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {label:30s} → {score:.2f}")

    print("\n✅ Persona B mastery extraction complete!")
    return extracted


def main():
    print("=" * 60)
    print("MASTERY SCORING TEST — Profile Classification + Bloom Decay")
    print("=" * 60)

    print("\n── Testing Persona A (Senior Frontend Engineer) ──")
    test_persona_a_mastery()

    print("\n" + "=" * 60)
    print("\n── Testing Persona B (Warehouse Operations Supervisor) ──")
    test_persona_b_mastery()


if __name__ == "__main__":
    main()
