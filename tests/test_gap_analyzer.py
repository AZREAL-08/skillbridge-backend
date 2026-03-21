import networkx as nx
from app.pathing.gap_analyzer import compute_skill_gap, get_active_subgraph, NOISE_TAXONOMY_IDS
from app.state import SkillEntry
from app.extractor.extractor import extract_skills, extract_skills_from_jd
from app.catalog.loader import load_catalog
from app.pathing.dag_builder import build_dag
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _run_gap_analyzer_from_persona_and_catalog(output_dir: Path) -> dict:
    """Run extractor -> persist -> reload -> gap analyzer using catalog DAG.
    
    Always regenerates processed files to avoid stale cache.
    Uses extract_skills_from_jd() for JD text (no mastery scoring).
    """
    project_root = Path(__file__).resolve().parents[1]
    persona_resume_path = project_root / "data" / "persona_a_resume.txt"
    persona_jd_path = project_root / "data" / "persona_a_jd.txt"

    resume_text = persona_resume_path.read_text(encoding="utf-8")
    jd_text = persona_jd_path.read_text(encoding="utf-8")

    catalog = load_catalog()
    assert catalog, "Catalog must be available at data/catalog.json"

    # Extract resume skills WITH mastery (profile classification + bloom decay)
    extracted_resume = extract_skills(resume_text, catalog)
    # Extract JD skills WITHOUT mastery
    extracted_jd = extract_skills_from_jd(jd_text)

    assert extracted_resume, "Extractor should return at least one resume skill"
    assert extracted_jd, "Extractor should return at least one JD skill"

    output_dir.mkdir(parents=True, exist_ok=True)

    persisted_resume = output_dir / "persona_a_extracted_resume.json"
    persisted_resume.write_text(json.dumps(extracted_resume, indent=2), encoding="utf-8")

    persisted_jd = output_dir / "persona_a_extracted_jd.json"
    persisted_jd.write_text(json.dumps(extracted_jd, indent=2), encoding="utf-8")

    loaded_extracted_resume = json.loads(persisted_resume.read_text(encoding="utf-8"))
    required_skills = [s["taxonomy_id"] for s in extracted_jd if s.get("taxonomy_source") == "emsi"]

    assert required_skills, "Required JD skills must include emsi taxonomy IDs"

    skill_gap = compute_skill_gap(loaded_extracted_resume, required_skills)

    G = build_dag(catalog)
    node_states = get_active_subgraph(G, skill_gap, loaded_extracted_resume)

    # Save diff output
    diff_file = output_dir / "persona_a_diff_output.json"
    diff_payload = {
        "required_skills": required_skills,
        "required_skills_after_noise_filter": [
            s for s in required_skills if s not in NOISE_TAXONOMY_IDS
        ],
        "skill_gap": skill_gap,
        "counts": {
            "required_raw": len(required_skills),
            "required_filtered": len([s for s in required_skills if s not in NOISE_TAXONOMY_IDS]),
            "gap": len(skill_gap),
        },
    }
    diff_file.write_text(json.dumps(diff_payload, indent=2), encoding="utf-8")

    return {
        "persisted_resume_file": str(persisted_resume),
        "persisted_jd_file": str(persisted_jd),
        "resume_skill_count": len(loaded_extracted_resume),
        "required_skill_count": len(required_skills),
        "skill_gap_count": len(skill_gap),
        "skill_gap": skill_gap,
        "node_states": node_states,
    }

def test_compute_skill_gap():
    extracted = [
        {"taxonomy_id": "s1", "taxonomy_source": "emsi", "label": "L1", "mastery_score": 0.9, "confidence_score": 1.0},
        {"taxonomy_id": "s2", "taxonomy_source": "emsi", "label": "L2", "mastery_score": 0.5, "confidence_score": 1.0},
    ]
    required = ["s1", "s2", "s3"]
    # s1 is mastered (0.9 >= 0.85), so gap should be s2, s3
    gap = compute_skill_gap(extracted, required)
    assert set(gap) == {"s2", "s3"}

def test_compute_skill_gap_inferred_excluded():
    """Inferred skills should NOT count as mastered for gap analysis."""
    extracted = [
        {"taxonomy_id": "s1", "taxonomy_source": "inferred", "label": "L1", "mastery_score": 0.9, "confidence_score": 1.0},
        {"taxonomy_id": "s2", "taxonomy_source": "emsi", "label": "L2", "mastery_score": 0.5, "confidence_score": 1.0},
    ]
    required = ["s1", "s2", "s3"]
    # s1 is inferred (not emsi), so it doesn't count as mastered
    gap = compute_skill_gap(extracted, required)
    assert set(gap) == {"s1", "s2", "s3"}

def test_noise_skill_filtering():
    """Noise skills should be filtered from required_skills before gap analysis."""
    extracted = [
        {"taxonomy_id": "s1", "taxonomy_source": "emsi", "label": "L1", "mastery_score": 0.5, "confidence_score": 1.0},
    ]
    # Include real skill + known noise IDs
    required = [
        "s1",
        "KS4425C7820LCHZS7VGX",  # "write" — noise
        "KS4400B70JFSWXTYH0P2",  # "nice" — noise
        "KS124RX787SQ1WVD8XF6",  # "scalable" — noise
    ]
    gap = compute_skill_gap(extracted, required)
    # Only s1 should remain (noise filtered, s1 not mastered at 0.5)
    assert gap == ["s1"], f"Expected ['s1'], got {gap}"

def test_get_active_subgraph_with_prereq():
    G = nx.DiGraph()
    # A teaches s1, B teaches s2, B is prereq for A
    G.add_node("A", skills_taught=["s1"], prerequisites=["B"], title="Course A")
    G.add_node("B", skills_taught=["s2"], prerequisites=[], title="Course B")
    G.add_edge("B", "A")
    
    skill_gap = ["s1"]
    extracted = [
        {"taxonomy_id": "s1", "taxonomy_source": "emsi", "label": "L1", "mastery_score": 0.0, "confidence_score": 1.0},
        {"taxonomy_id": "s2", "taxonomy_source": "emsi", "label": "L2", "mastery_score": 0.0, "confidence_score": 1.0},
    ]
    
    states = get_active_subgraph(G, skill_gap, extracted)
    # A should be assigned, B should be prerequisite
    assert states["A"] == "assigned"
    assert states["B"] == "prerequisite"

def test_get_active_subgraph_skipped():
    G = nx.DiGraph()
    G.add_node("A", skills_taught=["s1"], prerequisites=["B"], title="Course A")
    G.add_node("B", skills_taught=["s2"], prerequisites=[], title="Course B")
    G.add_edge("B", "A")
    
    skill_gap = ["s1"]
    extracted = [
        {"taxonomy_id": "s1", "taxonomy_source": "emsi", "label": "L1", "mastery_score": 0.0, "confidence_score": 1.0},
        {"taxonomy_id": "s2", "taxonomy_source": "emsi", "label": "L2", "mastery_score": 0.9, "confidence_score": 1.0},
    ]
    
    states = get_active_subgraph(G, skill_gap, extracted)
    # A should be assigned, B should be skipped
    assert states["A"] == "assigned"
    assert states["B"] == "skipped"


def test_gap_analyzer_from_extractor_output_and_catalog(tmp_path):
    """Full flow: extractor -> save -> reload -> gap analyzer using catalog DAG."""
    result = _run_gap_analyzer_from_persona_and_catalog(tmp_path)
    assert isinstance(result["node_states"], dict)
    if result["skill_gap"]:
        assert any(state in ("assigned", "prerequisite", "skipped") for state in result["node_states"].values())


def main():
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "processed"
    result = _run_gap_analyzer_from_persona_and_catalog(output_dir)

    print("Gap analyzer run completed")
    print(f"Persisted extracted resume: {result['persisted_resume_file']}")
    print(f"Persisted extracted jd: {result['persisted_jd_file']}")
    print(f"Resume skills: {result['resume_skill_count']}")
    print(f"Required JD skills: {result['required_skill_count']}")
    print(f"Skill gap count: {result['skill_gap_count']}")
    print(f"Skill gap ids: {result['skill_gap']}")
    print(f"Node states: {result['node_states']}")


if __name__ == "__main__":
    main()
