import json
import os
from typing import List, Dict, Any

def load_catalog(file_path: str = "data/catalog.json") -> List[Dict[str, Any]]:
    """Loads the catalog from a JSON file."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def validate_catalog(catalog: List[Dict[str, Any]]):
    """Validates basic catalog integrity: unique IDs and existing prerequisites."""
    # Check for unique IDs
    ids = [course.get('course_id') for course in catalog if 'course_id' in course]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate course_ids found in catalog")
    
    # Check for missing prerequisites
    all_ids = set(ids)
    for course in catalog:
        for prereq in course.get('prerequisites', []):
            if prereq not in all_ids:
                raise ValueError(f"Course {course['course_id']} has missing prerequisite: {prereq}")
