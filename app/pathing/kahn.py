"""
Kahn's algorithm with priority queue.
Max-Heap pops the course with the highest competency gain per hour.
"""

import heapq


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


def kahn_sort(zero_indegree_nodes: list) -> list:
    """
    Topological sort using Kahn's algorithm with a max-heap.
    Each node must expose: .gap_count, .mastery, .hours, and an id.
    """
    heap: list = []
    for node in zero_indegree_nodes:
        priority = compute_priority(node.gap_count, node.mastery, node.hours)
        heapq.heappush(heap, (-priority, node))  # negative for max-heap

    ordered: list = []
    while heap:
        _, node = heapq.heappop(heap)
        ordered.append(node)
        # TODO: decrement in-degree of dependents and push new zero-indegree nodes

    return ordered
