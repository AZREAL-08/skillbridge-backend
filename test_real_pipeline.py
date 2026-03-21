#!/usr/bin/env python
"""
Real-world pipeline test script.
Usage:
  uv run python test_real_pipeline.py --resume data/persona_a_resume.txt --jd data/persona_a_jd.txt --domain technical
"""

import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from app.extractor.extractor import extract_skills, extract_skills_from_jd
from app.pathing.pathing import add_skipped_nodes
from app.pathing.gap_analyzer import compute_skill_gap, get_active_subgraph
from app.catalog.loader import load_catalog
from app.pathing.dag_builder import build_dag
from app.pathing.kahn import kahn_priority_sort

# Load environment variables (GROQ_API_KEY)
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Test SkillBridge pipeline with real files."
    )
    parser.add_argument(
        "--resume", type=str, required=True, help="Path to resume text file"
    )
    parser.add_argument("--jd", type=str, required=True, help="Path to JD text file")
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=["technical", "operations"],
        help="Domain filter for course assignment",
    )

    args = parser.parse_args()

    # 1. Validate and Read Files
    resume_path = Path(args.resume)
    jd_path = Path(args.jd)

    if not resume_path.exists():
        print(f"Error: Resume file not found at {resume_path}")
        return
    if not jd_path.exists():
        print(f"Error: JD file not found at {jd_path}")
        return

    resume_text = resume_path.read_text(encoding="utf-8")
    jd_text = jd_path.read_text(encoding="utf-8")

    # 2. Load Catalog
    print("--- Loading Catalog ---")
    catalog = load_catalog()
    G = build_dag(catalog)
    print(f"Loaded {len(catalog)} courses.")

    # 3. Extraction (REAL NLP)
    print("\n--- Extracting Skills (Real NLP Pipeline) ---")
    print("Extracting from Resume...")
    current_skills = extract_skills(resume_text, catalog)
    print(f"Found {len(current_skills)} skills in resume.")

    print("Extracting from JD...")
    target_skill_entries = extract_skills_from_jd(jd_text)
    required_ids = [
        s["taxonomy_id"]
        for s in target_skill_entries
        if s.get("taxonomy_source") == "emsi"
    ]
    print(f"Found {len(required_ids)} required skills in JD.")

    # 4. Gap Analysis
    print(f"\n--- Analyzing Gap (Domain: {args.domain or 'None'}) ---")
    gap = compute_skill_gap(current_skills, required_ids, catalog, args.domain)
    node_states = get_active_subgraph(G, gap, current_skills, args.domain)

    # 5. Pathway Generation (Kahn Sort)
    print("\n--- Generating Pathway ---")
    active_ids = list(node_states.keys())

    # Precompute metadata for Kahn
    gap_set = set(gap)
    course_metadata = {}
    for cid in active_ids:
        meta = G.nodes[cid]
        taught = meta.get("skills_taught", [])
        gap_count = len([s for s in taught if s in gap_set])

        masteries = []
        for tid in taught:
            m = 0.0
            for s in current_skills:
                if s["taxonomy_id"] == tid:
                    m = s["mastery_score"]
                    break
            masteries.append(m)
        avg_mastery = sum(masteries) / len(masteries) if masteries else 0.0

        course_metadata[cid] = {
            "prerequisites": meta.get("prerequisites", []),
            "gap_count": gap_count,
            "mastery": avg_mastery,
            "hours": meta.get("estimated_hours", 1.0),
            "bloom_level": meta.get("bloom_level", 3),
        }

    sorted_ids = kahn_priority_sort(active_ids, course_metadata)

    # Build initial pathway
    initial_pathway = []
    for cid in sorted_ids:
        initial_pathway.append(
            {
                "course_id": cid,
                "node_state": node_states[cid],
                "mastery_score": course_metadata[cid]["mastery"],
                "confidence_score": 0.9,
            }
        )

    # Add independent skipped nodes
    final_pathway = add_skipped_nodes(
        initial_pathway, current_skills, required_ids, catalog
    )

    # 6. Final Report
    print("\n" + "=" * 60)
    print("FINAL PATHWAY REPORT")
    print("=" * 60)

    print("\n[ SKILL GAP ]")
    # Map IDs to labels for readability
    id_to_label = {s["taxonomy_id"]: s["label"] for s in current_skills}
    id_to_label.update({s["taxonomy_id"]: s["label"] for s in target_skill_entries})

    for gid in gap:
        label = id_to_label.get(gid, gid)
        print(f" - {label}")

    print("\n[ RECOMMENDED COURSES ]")
    for item in final_pathway:
        cid = item["course_id"]
        state = item["node_state"]
        if state != "skipped":
            title = G.nodes[cid].get("title", cid)
            print(f" -> [{state.upper()}] {cid}: {title}")

    print("\n[ SKIPPED COURSES ]")
    for item in final_pathway:
        cid = item["course_id"]
        state = item["node_state"]
        if state == "skipped":
            # Check if it's in catalog or G
            title = G.nodes[cid].get("title", cid) if cid in G else cid
            print(
                f" -- [SKIPPED] {cid}: {title} (Mastery: {item['mastery_score']:.2f})"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
