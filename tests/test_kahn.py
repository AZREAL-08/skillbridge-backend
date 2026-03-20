from app.pathing.kahn import kahn_priority_sort

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
