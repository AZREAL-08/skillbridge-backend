from app.pathing.kahn import kahn_priority_sort
from app.pathing.gap_analyzer import compute_skill_gap, get_active_subgraph
from app.catalog.loader import load_catalog
from app.pathing.dag_builder import build_dag
from app.extractor.extractor import extract_skills
from pathlib import Path
import json

def test_kahn_priority():
    # A -> B
    # A -> C
    # B and C are zero-indegree after A. C has higher priority.
    # Priority formula: (gap_count * (1 - mastery)) / (hours + 1e-5)
    nodes = ['A', 'B', 'C']
    meta = {
        'A': {'prerequisites': [], 'gap_count': 1, 'mastery': 0.0, 'hours': 1.0}, # Priority ~ 1.0
        'B': {'prerequisites': ['A'], 'gap_count': 1, 'mastery': 0.0, 'hours': 10.0}, # Priority ~ 0.1
        'C': {'prerequisites': ['A'], 'gap_count': 1, 'mastery': 0.0, 'hours': 1.0}, # Priority ~ 1.0
    }
    # To ensure C > B in priority:
    # B priority: (1 * 1) / 10 = 0.1
    # C priority: (1 * 1) / 1 = 1.0
    order = kahn_priority_sort(nodes, meta)
    assert order == ['A', 'C', 'B']

def test_kahn_independent():
    # A and B are independent. B has higher priority.
    nodes = ['A', 'B']
    meta = {
        'A': {'prerequisites': [], 'gap_count': 1, 'mastery': 0.0, 'hours': 10.0}, # Priority ~ 0.1
        'B': {'prerequisites': [], 'gap_count': 1, 'mastery': 0.0, 'hours': 1.0}, # Priority ~ 1.0
    }
    order = kahn_priority_sort(nodes, meta)
    assert order == ['B', 'A']


def _load_or_build_processed_inputs(output_dir: Path) -> tuple[list[dict], list[dict], list[str], Path]:
    """Load processed artifacts; create them if missing, and persist diff output."""
    project_root = Path(__file__).resolve().parents[1]
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_file = output_dir / "persona_a_extracted_resume.json"
    jd_file = output_dir / "persona_a_extracted_jd.json"
    diff_file = output_dir / "persona_a_diff_output.json"

    if resume_file.exists():
        extracted_resume = json.loads(resume_file.read_text(encoding="utf-8"))
    else:
        resume_text = (project_root / "data" / "persona_a_resume.txt").read_text(encoding="utf-8")
        extracted_resume = extract_skills(resume_text)
        resume_file.write_text(json.dumps(extracted_resume, indent=2), encoding="utf-8")

    if jd_file.exists():
        extracted_jd = json.loads(jd_file.read_text(encoding="utf-8"))
    else:
        jd_text = (project_root / "data" / "persona_a_jd.txt").read_text(encoding="utf-8")
        extracted_jd = extract_skills(jd_text)
        jd_file.write_text(json.dumps(extracted_jd, indent=2), encoding="utf-8")

    assert extracted_resume, "Processed resume extraction must not be empty"
    assert extracted_jd, "Processed JD extraction must not be empty"

    jd_required = [s["taxonomy_id"] for s in extracted_jd if s.get("taxonomy_source") == "emsi"]
    assert jd_required, "Processed JD must contain emsi taxonomy IDs"

    if diff_file.exists():
        diff_payload = json.loads(diff_file.read_text(encoding="utf-8"))
        skill_gap = diff_payload.get("skill_gap", [])
    else:
        skill_gap = compute_skill_gap(extracted_resume, jd_required)
        diff_payload = {
            "required_skills": jd_required,
            "skill_gap": skill_gap,
            "counts": {
                "required": len(jd_required),
                "gap": len(skill_gap),
            },
        }
        diff_file.write_text(json.dumps(diff_payload, indent=2), encoding="utf-8")

    return extracted_resume, extracted_jd, skill_gap, diff_file


def _run_kahn_from_catalog_and_processed(output_dir: Path) -> dict:
    """Build Kahn input from catalog + extracted JD + diff output, then persist result."""
    extracted_resume, extracted_jd, skill_gap, diff_file = _load_or_build_processed_inputs(output_dir)

    catalog = load_catalog()
    assert catalog, "Catalog must be available in data/catalog.json"
    G = build_dag(catalog)

    node_states = get_active_subgraph(G, skill_gap, extracted_resume)
    active_node_ids = list(node_states.keys())

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
        }

    kahn_order = kahn_priority_sort(active_node_ids, course_metadata) if active_node_ids else []

    kahn_file = output_dir / "persona_a_kahn.json"
    kahn_payload = {
        "input_diff_file": str(diff_file),
        "active_node_ids": active_node_ids,
        "node_states": node_states,
        "kahn_order": kahn_order,
    }
    kahn_file.write_text(json.dumps(kahn_payload, indent=2), encoding="utf-8")

    return {
        "diff_file": str(diff_file),
        "kahn_file": str(kahn_file),
        "kahn_order": kahn_order,
        "active_node_ids": active_node_ids,
    }


def test_kahn_from_catalog_and_processed_artifacts():
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "processed"
    result = _run_kahn_from_catalog_and_processed(output_dir)

    assert Path(result["diff_file"]).exists()
    assert Path(result["kahn_file"]).exists()
    assert isinstance(result["kahn_order"], list)
    assert set(result["kahn_order"]).issubset(set(result["active_node_ids"]))


def main():
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "processed"
    result = _run_kahn_from_catalog_and_processed(output_dir)
    print("Kahn run completed")
    print(f"Saved diff output: {result['diff_file']}")
    print(f"Saved kahn output: {result['kahn_file']}")
    print(f"Active nodes: {len(result['active_node_ids'])}")
    print(f"Kahn order: {result['kahn_order']}")


if __name__ == "__main__":
    main()
