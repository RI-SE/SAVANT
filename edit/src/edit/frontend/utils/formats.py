from typing import Dict


def format_object_identity(identity: Dict) -> str:
    """Formats an object's identity into a standardized string."""
    obj_type = identity.get("type", "Unknown")
    obj_id = identity.get("id", "N/A")
    obj_name = identity.get("name", "")
    return f"{obj_type} (ID: {obj_id}) - {obj_name}"
