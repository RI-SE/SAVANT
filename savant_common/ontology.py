import re
from typing import Optional

from rdflib import Graph, Namespace, RDF, RDFS


def read_ontology_classes(ttl_path):
    """
    Reads a Turtle ontology file and returns a list of classes with their labels and UIDs.

    Returns a list of dictionaries with:
    - uri: Full URI of the class
    - class_name: Short class name (extracted from URI)
    - label: Human-readable label
    - uid: Unique identifier
    - parent_class: Direct parent class URI (or None)
    - parent_class_name: Direct parent class name (or None)
    - top_level_class: Root class in the inheritance chain URI (or None)
    - top_level_class_name: Root class name (or None)
    """
    g = Graph()
    g.parse(ttl_path, format="turtle")

    SAVANT = Namespace("https://ri-se.github.io/SAVANT/ontology/savant#")

    def extract_class_name(uri_str):
        """Extract the class name from a URI (part after #)."""
        if '#' in uri_str:
            return uri_str.split('#')[-1]
        elif '/' in uri_str:
            return uri_str.split('/')[-1]
        return uri_str

    def find_top_level_class(class_uri, visited=None):
        """
        Recursively find the top-level class (root of inheritance chain).

        A top-level class is one that either:
        1. Has no rdfs:subClassOf relationship, or
        2. Has rdfs:subClassOf pointing to a class outside the SAVANT namespace

        Returns tuple: (top_level_uri, top_level_name) or (None, None)
        """
        if visited is None:
            visited = set()

        # Prevent infinite loops
        if class_uri in visited:
            return None, None
        visited.add(class_uri)

        # Get parent class
        parent = g.value(class_uri, RDFS.subClassOf)

        # If no parent, this is the top-level class
        if parent is None:
            return str(class_uri), extract_class_name(str(class_uri))

        # If parent is outside SAVANT namespace, current class is top-level
        parent_str = str(parent)
        if not parent_str.startswith("https://ri-se.github.io/SAVANT/ontology/savant#"):
            return str(class_uri), extract_class_name(str(class_uri))

        # Otherwise, recurse up the chain
        return find_top_level_class(parent, visited)

    classes = []

    for s in g.subjects(RDF.type, RDFS.Class):
        uri_str = str(s)
        label = g.value(s, RDFS.label)
        uid = g.value(s, SAVANT.uid)

        # Get direct parent class
        parent = g.value(s, RDFS.subClassOf)
        parent_uri = str(parent) if parent else None
        parent_name = extract_class_name(parent_uri) if parent_uri else None

        # Find top-level class in the inheritance chain
        top_level_uri, top_level_name = find_top_level_class(s)

        classes.append({
            "uri": uri_str,
            "class_name": extract_class_name(uri_str),
            "label": str(label) if label else None,
            "uid": int(uid) if uid is not None else None,
            "parent_class": parent_uri,
            "parent_class_name": parent_name,
            "top_level_class": top_level_uri,
            "top_level_class_name": top_level_name
        })

    return classes


def create_class_map(ttl_path, filter_by_uid=None, filter_by_top_level=None):
    """
    Create a class_id → label mapping from ontology.

    This is useful for YOLO applications where you need a dictionary mapping
    class IDs to human-readable labels for annotation and visualization.

    Args:
        ttl_path: Path to the Turtle ontology file (.ttl)
        filter_by_uid: Optional list/set of UIDs to include (e.g., [110, 111, 113, 117])
        filter_by_top_level: Optional top-level class name to filter by (e.g., "RoadUserVehicle")

    Returns:
        Dictionary mapping UID → label (e.g., {110: "Car", 111: "Truck", ...})
        Only includes classes with valid (non-None) UIDs.

    Example:
        # Get all classes with UIDs
        class_map = create_class_map("savant_ontology_1.3.0.ttl")

        # Get only specific vehicle classes
        class_map = create_class_map("savant_ontology_1.3.0.ttl",
                                     filter_by_uid=[110, 111, 113, 117])

        # Get all vehicle classes
        class_map = create_class_map("savant_ontology_1.3.0.ttl",
                                     filter_by_top_level="RoadUserVehicle")
    """
    classes = read_ontology_classes(ttl_path)

    # Filter by UID if specified
    if filter_by_uid is not None:
        filter_set = set(filter_by_uid)
        classes = [c for c in classes if c['uid'] in filter_set]

    # Filter by top-level class if specified
    if filter_by_top_level is not None:
        classes = [c for c in classes if c['top_level_class_name'] == filter_by_top_level]

    # Create mapping, excluding classes without UIDs
    return {c['uid']: c['label'] for c in classes if c['uid'] is not None}


def get_classes_by_category(ttl_path, top_level_class_name):
    """
    Get all classes that belong to a specific top-level category.

    Args:
        ttl_path: Path to the Turtle ontology file (.ttl)
        top_level_class_name: Name of the top-level class (e.g., "RoadUserVehicle", "DynamicObject")

    Returns:
        List of class dictionaries (same format as read_ontology_classes)
        Sorted by UID (None UIDs at the end)

    Example:
        # Get all vehicle classes
        vehicles = get_classes_by_category("savant_ontology_1.3.0.ttl", "RoadUserVehicle")
        for v in vehicles:
            print(f"{v['uid']}: {v['label']} ({v['class_name']})")
    """
    classes = read_ontology_classes(ttl_path)
    filtered = [c for c in classes if c['top_level_class_name'] == top_level_class_name]

    # Sort by UID, with None values at the end
    return sorted(filtered, key=lambda x: (x['uid'] is None, x['uid']))


def get_class_by_uid(ttl_path, uid):
    """
    Get a specific class by its UID.

    Args:
        ttl_path: Path to the Turtle ontology file (.ttl)
        uid: The UID to look up (integer)

    Returns:
        Class dictionary if found, None otherwise

    Example:
        # Look up class with UID 110
        cls = get_class_by_uid("savant_ontology_1.3.0.ttl", 110)
        if cls:
            print(f"UID {uid}: {cls['label']} ({cls['class_name']})")
    """
    classes = read_ontology_classes(ttl_path)
    for c in classes:
        if c['uid'] == uid:
            return c
    return None


def get_class_by_label(ttl_path, label, case_sensitive=False):
    """
    Get a class by its label.

    Args:
        ttl_path: Path to the Turtle ontology file (.ttl)
        label: The label to look up (string)
        case_sensitive: Whether to do case-sensitive matching (default: False)

    Returns:
        Class dictionary if found, None otherwise (returns first match if multiple)

    Example:
        # Look up by label
        cls = get_class_by_label("savant_ontology_1.3.0.ttl", "car")
        if cls:
            print(f"{cls['label']} has UID {cls['uid']}")
    """
    classes = read_ontology_classes(ttl_path)

    if case_sensitive:
        for c in classes:
            if c['label'] == label:
                return c
    else:
        label_lower = label.lower() if label else None
        for c in classes:
            if c['label'] and c['label'].lower() == label_lower:
                return c

    return None


def extract_namespace_uri(ttl_path: str) -> Optional[str]:
    """
    Extract the default namespace URI from a Turtle ontology file.

    Parses the @prefix : <URI> . declaration to get the default namespace.

    Args:
        ttl_path: Path to the Turtle ontology file (.ttl)

    Returns:
        The namespace URI string if found, None otherwise.

    Example:
        # For a file with "@prefix : <https://ri-se.github.io/SAVANT/ontology/savant#> ."
        uri = extract_namespace_uri("savant_ontology_1.3.1.ttl")
        # Returns: "https://ri-se.github.io/SAVANT/ontology/savant#"
    """
    with open(ttl_path, 'r') as f:
        for line in f:
            match = re.match(r'@prefix\s+:\s+<(.+)>\s*\.', line)
            if match:
                return match.group(1)
    return None


# Example usage:
# classes = read_ontology_classes("savant_ontology_1.0.0.ttl")
# for c in classes:
#     print(c)
#
# class_map = create_class_map("savant_ontology_1.3.0.ttl", filter_by_uid=[110, 111, 113, 117])
# vehicles = get_classes_by_category("savant_ontology_1.3.0.ttl", "RoadUserVehicle")
# car_class = get_class_by_uid("savant_ontology_1.3.0.ttl", 110)
