from dataclasses import dataclass


@dataclass
class SidebarState:
    """Dataclass to hold the state of the sidebar."""

    historic_obj_frame_count: int = (
        50  # Number of previous frames to consider for recent objects
    )
