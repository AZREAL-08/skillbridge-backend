import networkx as nx
from typing import List, Dict, Any
from app.catalog.loader import load_catalog, validate_catalog

def build_dag(catalog: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Builds a NetworkX DAG from the catalog.
    Validates unique IDs, missing prerequisites, and cycles.
    :param catalog: List of course records
    :return: A NetworkX DiGraph where each node has course metadata.
    """
    validate_catalog(catalog)
    
    G = nx.DiGraph()
    for course in catalog:
        cid = course['course_id']
        # Add course node with all metadata
        G.add_node(cid, **course)
        # Add edges from prerequisites to this course
        for prereq in course.get('prerequisites', []):
            G.add_edge(prereq, cid)
            
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Cycle detected in the catalog graph")
        
    return G
