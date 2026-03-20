import networkx as nx
from app.pathing.gap_analyzer import get_skill_gap, get_active_subgraph
from app.state import SkillEntry

def test_get_skill_gap():
    extracted = [
        {"esco_uri": "s1", "label": "L1", "mastery_score": 0.9, "confidence_score": 1.0},
        {"esco_uri": "s2", "label": "L2", "mastery_score": 0.5, "confidence_score": 1.0},
    ]
    required = ["s1", "s2", "s3"]
    # s1 is mastered (0.9 >= 0.85), so gap should be s2, s3
    gap = get_skill_gap(extracted, required)
    assert set(gap) == {"s2", "s3"}

def test_get_active_subgraph_with_prereq():
    G = nx.DiGraph()
    # A teaches s1, B teaches s2, B is prereq for A
    G.add_node("A", skills_taught=["s1"], prerequisites=["B"], title="Course A")
    G.add_node("B", skills_taught=["s2"], prerequisites=[], title="Course B")
    G.add_edge("B", "A")
    
    skill_gap = ["s1"]
    extracted = [
        {"esco_uri": "s1", "label": "L1", "mastery_score": 0.0, "confidence_score": 1.0},
        {"esco_uri": "s2", "label": "L2", "mastery_score": 0.0, "confidence_score": 1.0},
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
        {"esco_uri": "s1", "label": "L1", "mastery_score": 0.0, "confidence_score": 1.0},
        {"esco_uri": "s2", "label": "L2", "mastery_score": 0.9, "confidence_score": 1.0},
    ]
    
    states = get_active_subgraph(G, skill_gap, extracted)
    # A should be assigned, B should be skipped
    assert states["A"] == "assigned"
    assert states["B"] == "skipped"
