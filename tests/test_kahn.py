from app.pathing.kahn import kahn_priority_sort

def test_kahn_priority():
    # A -> B
    # A -> C
    # B and C are zero-indegree after A. C has higher priority.
    nodes = ['A', 'B', 'C']
    meta = {
        'A': {'prerequisites': [], 'priority': 10},
        'B': {'prerequisites': ['A'], 'priority': 5},
        'C': {'prerequisites': ['A'], 'priority': 20},
    }
    order = kahn_priority_sort(nodes, meta)
    assert order == ['A', 'C', 'B']

def test_kahn_independent():
    # A and B are independent. B has higher priority.
    nodes = ['A', 'B']
    meta = {
        'A': {'prerequisites': [], 'priority': 10},
        'B': {'prerequisites': [], 'priority': 20},
    }
    order = kahn_priority_sort(nodes, meta)
    assert order == ['B', 'A']
