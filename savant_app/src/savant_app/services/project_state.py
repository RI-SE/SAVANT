"""Class representing and managing the complete state of an annotation project."""
import json
from savant_app.models.OpenLabel import OpenLabel 
from savant_app.utils import read_json


class ProjectState:
    def __init__(self):
        self.annotation_config: OpenLabel = None
        self.open_label_path: str = None

    def load_openlabel_config(self, path: str) -> None:
        """Load and validate OpenLabel configuration from JSON file.
        Args:
            path: Path to JSON file containing a SAVANT OpenLabel configuration

        Raises:
            FileNotFoundError: If specified path doesn't exist
            ValidationError: If configuration fails OpenLabel schema validation
            ValueError: If path does not point to a JSON file.

        Initializes:
            self.open_label: New OpenLabel instance with loaded configuration
        """
        config = read_json(path)
        self.annotation_config = OpenLabel(**config["openlabel"])
        self.open_label_path = path

    def save_openlabel_config(self) -> None:
        """Save the adjusted OpenLabel configuration to a JSON file.

        Args:
            adjusted_config: The OpenLabel instance containing the adjusted configuration

        Raises:
            ValueError: If adjusted_config is not a valid OpenLabel instance
        """
        # Save the configuration to a JSON file
        with open(self.open_label_path, "w") as f:
            f.write(json.dumps({"openlabel": self.annotation_config.model_dump(mode="json")}))
