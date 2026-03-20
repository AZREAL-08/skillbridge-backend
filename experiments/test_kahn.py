import heapq

def kahn_priority_sort(active_subgraph_nodes, course_metadata):
    """
    Topologically sorts nodes in the active subgraph using priority-based Kahn's algorithm.
    :param active_subgraph_nodes: list of course_ids in the subgraph
    :param course_metadata: dict mapping course_id to metadata (prerequisites, priority)
    :return: list of course_ids in sorted order
    """
    # Build local graph for the subgraph
    local_adj = {cid: [] for cid in active_subgraph_nodes}
    local_indegree = {cid: 0 for cid in active_subgraph_nodes}
    
    for cid in active_subgraph_nodes:
        meta = course_metadata[cid]
        for prereq in meta.get('prerequisites', []):
            if prereq in local_adj: # Only care about prerequisites in the subgraph
                local_adj[prereq].append(cid)
                local_indegree[cid] += 1
                
    # Max-heap for priority (use negative values for heapq)
    # priority = (skill_gap_count * (1 - mastery_score)) / (estimated_hours + 1e-5)
    priority_queue = []
    for cid in active_subgraph_nodes:
        if local_indegree[cid] == 0:
            heapq.heappush(priority_queue, (-course_metadata[cid]['priority'], cid))
            
    sorted_order = []
    while priority_queue:
        neg_priority, current_cid = heapq.heappop(priority_queue)
        sorted_order.append(current_cid)
        
        for neighbor in local_adj[current_cid]:
            local_indegree[neighbor] -= 1
            if local_indegree[neighbor] == 0:
                heapq.heappush(priority_queue, (-course_metadata[neighbor]['priority'], neighbor))
                
    if len(sorted_order) != len(active_subgraph_nodes):
        raise ValueError("Cycle detected in the active subgraph")
        
    return sorted_order

# Test
if __name__ == "__main__":
    # Toy DAG
    # A -> B -> D
    # A -> C -> D
    # E (independent)
    nodes = ['A', 'B', 'C', 'D', 'E']
    metadata = {
        'A': {'prerequisites': [], 'priority': 10},
        'B': {'prerequisites': ['A'], 'priority': 5},
        'C': {'prerequisites': ['A'], 'priority': 20},
        'D': {'prerequisites': ['B', 'C'], 'priority': 50},
        'E': {'prerequisites': [], 'priority': 100},
    }
    
    # Expected: E (priority 100) and A (priority 10) are zero-indegree.
    # E should be first.
    # Then A.
    # After A, B (priority 5) and C (priority 20) become zero-indegree.
    # C should be next.
    # Then B.
    # Finally D.
    # Order: E, A, C, B, D
    
    try:
        result = kahn_priority_sort(nodes, metadata)
        print(f"Result: {result}")
        assert result == ['E', 'A', 'C', 'B', 'D']
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        exit(1)
