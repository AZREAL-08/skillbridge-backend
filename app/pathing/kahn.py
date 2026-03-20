"""
Kahn's algorithm with priority queue.
Max-Heap pops the course with the highest competency gain per hour.
"""

import heapq
from typing import List, Dict, Any

def compute_priority(
    skill_gap_count: int,
    mastery_score: float,
    estimated_hours: float,
) -> float:
    """
    Priority = (gap_count * (1 - mastery)) / hours.
    Higher is better — courses that close the most gaps cheapest win.
    """
    return (skill_gap_count * (1 - mastery_score)) / (estimated_hours + 1e-5)

def kahn_priority_sort(active_subgraph_nodes: List[str], course_metadata: Dict[str, Any]) -> List[str]:
    """
    Topologically sorts nodes in the active subgraph using priority-based Kahn's algorithm.
    :param active_subgraph_nodes: list of course_ids in the subgraph
    :param course_metadata: dict mapping course_id to metadata (prerequisites, skills_taught, mastery, hours)
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
    priority_queue = []
    for cid in active_subgraph_nodes:
        if local_indegree[cid] == 0:
            meta = course_metadata[cid]
            # Priority components from metadata
            gap_count = meta.get('gap_count', 0)
            mastery = meta.get('mastery', 0.0)
            hours = meta.get('hours', 1.0)
            
            priority = compute_priority(gap_count, mastery, hours)
            heapq.heappush(priority_queue, (-priority, cid))
            
    sorted_order = []
    while priority_queue:
        neg_priority, current_cid = heapq.heappop(priority_queue)
        sorted_order.append(current_cid)
        
        for neighbor in local_adj[current_cid]:
            local_indegree[neighbor] -= 1
            if local_indegree[neighbor] == 0:
                meta = course_metadata[neighbor]
                gap_count = meta.get('gap_count', 0)
                mastery = meta.get('mastery', 0.0)
                hours = meta.get('hours', 1.0)
                
                priority = compute_priority(gap_count, mastery, hours)
                heapq.heappush(priority_queue, (-priority, neighbor))
                
    if len(sorted_order) != len(active_subgraph_nodes):
        raise ValueError("Cycle detected in the active subgraph")
        
    return sorted_order
