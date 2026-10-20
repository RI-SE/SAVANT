"""Resource path resolution for SAVANT package data.

Provides functions to locate data files (ontology, schema, YOLO weights) whether
SAVANT is installed as a package or running from the source repository.

Search order:
- Ontology/Schema: CLI arg → Package data → Relative path
- Weights: CLI arg → Relative path (never downloaded: .pt files are pickles that can
  run code, and a moving download source makes runs irreproducible)
"""

from pathlib import Path


def get_ontology_path() -> str:
    """Get ontology path from package data or fallback to relative path.

    Returns:
        Path to savant.ttl ontology file

    Raises:
        FileNotFoundError: If ontology file cannot be found
    """
    # 1. Try package data (installed via pip)
    try:
        from importlib.resources import files

        pkg_path = files("savant_common") / "ontology" / "savant.ttl"
        # Use is_file() for Traversable objects in Python 3.9+
        if hasattr(pkg_path, "is_file") and pkg_path.is_file():
            return str(pkg_path)
        # Fallback for older Python: try to access the file
        with pkg_path.open("r"):
            return str(pkg_path)
    except (ModuleNotFoundError, TypeError, FileNotFoundError, AttributeError):
        pass

    # 2. Try relative path (running from repo)
    repo_path = Path(__file__).parent.parent / "ontology" / "savant.ttl"
    if repo_path.exists():
        return str(repo_path)

    raise FileNotFoundError("Could not find savant.ttl ontology file")


def get_schema_path() -> str:
    """Get schema path from package data or fallback to relative path.

    Returns:
        Path to savant_openlabel_subset.schema.json file

    Raises:
        FileNotFoundError: If schema file cannot be found
    """
    schema_name = "savant_openlabel_subset.schema.json"

    # 1. Try package data
    try:
        from importlib.resources import files

        pkg_path = files("savant_common") / "schema" / schema_name
        if hasattr(pkg_path, "is_file") and pkg_path.is_file():
            return str(pkg_path)
        with pkg_path.open("r"):
            return str(pkg_path)
    except (ModuleNotFoundError, TypeError, FileNotFoundError, AttributeError):
        pass

    # 2. Try relative path
    repo_path = Path(__file__).parent.parent / "schema" / schema_name
    if repo_path.exists():
        return str(repo_path)

    raise FileNotFoundError(f"Could not find schema file: {schema_name}")


def get_weights_path(filename: str = "markit_yolo.pt") -> str:
    """Return the weights bundled with the source repository.

    Raises:
        FileNotFoundError: If not running from a repo checkout; pass --weights instead
    """
    repo_path = Path(__file__).parent.parent / "markit" / filename
    if repo_path.exists():
        return str(repo_path)
    raise FileNotFoundError(
        f"No YOLO weights found (looked for {repo_path}). "
        "Pass the weights file explicitly with --weights PATH."
    )
