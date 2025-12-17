"""Unit tests for ProjectService."""

import json
from pathlib import Path

import pytest

from trainit.trainit_gui.models.project import Project, TrainingConfig
from trainit.trainit_gui.services.project_service import ProjectService


class TestProjectServiceCreate:
    """Tests for project creation."""

    @pytest.fixture
    def service(self) -> ProjectService:
        return ProjectService()

    @pytest.mark.unit
    def test_create_project(
        self, service: ProjectService, temp_dir: Path, mock_datasets_root: Path
    ):
        """Test creating a new project."""
        project = service.create_project(
            name="New Project",
            project_folder=str(temp_dir),
            datasets_root=str(mock_datasets_root),
            description="Test description",
        )

        assert project.name == "New Project"
        assert project.datasets_root == str(mock_datasets_root)
        assert project.description == "Test description"

        project_file = temp_dir / "project.json"
        assert project_file.exists()

    @pytest.mark.unit
    def test_create_project_already_exists(
        self, service: ProjectService, temp_dir: Path, mock_datasets_root: Path
    ):
        """Test creating project when one already exists."""
        service.create_project("First", str(temp_dir), str(mock_datasets_root))

        with pytest.raises(FileExistsError):
            service.create_project("Second", str(temp_dir), str(mock_datasets_root))

    @pytest.mark.unit
    def test_create_project_invalid_folder(
        self, service: ProjectService, mock_datasets_root: Path
    ):
        """Test creating project with invalid folder."""
        with pytest.raises(ValueError, match="doesn't exist"):
            service.create_project(
                "Test", "/nonexistent/folder", str(mock_datasets_root)
            )


class TestProjectServiceLoad:
    """Tests for project loading."""

    @pytest.fixture
    def service(self) -> ProjectService:
        return ProjectService()

    @pytest.mark.unit
    def test_load_project(
        self, service: ProjectService, temp_dir: Path, sample_project_json: dict
    ):
        """Test loading an existing project."""
        project_file = temp_dir / "project.json"
        project_file.write_text(json.dumps(sample_project_json))

        project = service.load_project(str(project_file))

        assert project.name == "Test Project"
        assert len(project.configs) == 1

    @pytest.mark.unit
    def test_load_project_not_found(self, service: ProjectService):
        """Test loading nonexistent project."""
        with pytest.raises(FileNotFoundError):
            service.load_project("/nonexistent/project.json")

    @pytest.mark.unit
    def test_load_project_invalid_json(self, service: ProjectService, temp_dir: Path):
        """Test loading invalid JSON."""
        project_file = temp_dir / "project.json"
        project_file.write_text("not valid json {{{")

        with pytest.raises(ValueError, match="Invalid JSON"):
            service.load_project(str(project_file))


class TestProjectServiceSave:
    """Tests for project saving."""

    @pytest.fixture
    def service(self) -> ProjectService:
        return ProjectService()

    @pytest.mark.unit
    def test_save_project(
        self, service: ProjectService, temp_dir: Path, sample_project: Project
    ):
        """Test saving a project."""
        project_file = temp_dir / "project.json"

        service.save_project(sample_project, str(project_file))

        assert project_file.exists()

        with open(project_file) as f:
            data = json.load(f)
        assert data["name"] == "Test Project"

    @pytest.mark.unit
    def test_save_project_creates_parent_dirs(
        self, service: ProjectService, temp_dir: Path, sample_project: Project
    ):
        """Test that save creates parent directories."""
        project_file = temp_dir / "subdir" / "nested" / "project.json"

        service.save_project(sample_project, str(project_file))

        assert project_file.exists()

    @pytest.mark.unit
    def test_save_updates_modified_timestamp(
        self, service: ProjectService, temp_dir: Path, sample_project: Project
    ):
        """Test that save updates modified timestamp."""
        original_modified = sample_project.modified_at
        project_file = temp_dir / "project.json"

        service.save_project(sample_project, str(project_file))

        assert sample_project.modified_at != original_modified


class TestProjectServiceHelpers:
    """Tests for helper methods."""

    @pytest.fixture
    def service(self) -> ProjectService:
        return ProjectService()

    @pytest.mark.unit
    def test_get_project_file_path(self, service: ProjectService, temp_dir: Path):
        """Test getting project file path."""
        path = service.get_project_file_path(str(temp_dir))

        assert path == str(temp_dir / "project.json")

    @pytest.mark.unit
    def test_project_exists_true(self, service: ProjectService, temp_dir: Path):
        """Test project_exists returns True when file exists."""
        (temp_dir / "project.json").touch()

        assert service.project_exists(str(temp_dir)) is True

    @pytest.mark.unit
    def test_project_exists_false(self, service: ProjectService, temp_dir: Path):
        """Test project_exists returns False when file missing."""
        assert service.project_exists(str(temp_dir)) is False


class TestProjectServiceConfigOperations:
    """Tests for config-related operations."""

    @pytest.fixture
    def service(self) -> ProjectService:
        return ProjectService()

    @pytest.mark.unit
    def test_add_config_to_project(
        self, service: ProjectService, temp_dir: Path, sample_project_json: dict
    ):
        """Test adding a config to project."""
        project_file = temp_dir / "project.json"
        project_file.write_text(json.dumps(sample_project_json))

        new_config = TrainingConfig(name="new_config")
        project = service.add_config_to_project(str(project_file), new_config)

        assert len(project.configs) == 2
        assert project.get_config_by_name("new_config") is not None

    @pytest.mark.unit
    def test_add_duplicate_config_raises(
        self, service: ProjectService, temp_dir: Path, sample_project_json: dict
    ):
        """Test adding duplicate config raises error."""
        project_file = temp_dir / "project.json"
        project_file.write_text(json.dumps(sample_project_json))

        duplicate = TrainingConfig(name="config1")

        with pytest.raises(ValueError, match="already exists"):
            service.add_config_to_project(str(project_file), duplicate)

    @pytest.mark.unit
    def test_update_config_in_project(
        self, service: ProjectService, temp_dir: Path, sample_project_json: dict
    ):
        """Test updating a config in project."""
        project_file = temp_dir / "project.json"
        project_file.write_text(json.dumps(sample_project_json))

        updated_config = TrainingConfig(name="config1", epochs=200)
        project = service.update_config_in_project(str(project_file), updated_config)

        config = project.get_config_by_name("config1")
        assert config.epochs == 200

    @pytest.mark.unit
    def test_update_nonexistent_config_raises(
        self, service: ProjectService, temp_dir: Path, sample_project_json: dict
    ):
        """Test updating nonexistent config raises error."""
        project_file = temp_dir / "project.json"
        project_file.write_text(json.dumps(sample_project_json))

        config = TrainingConfig(name="nonexistent")

        with pytest.raises(ValueError, match="not found"):
            service.update_config_in_project(str(project_file), config)

    @pytest.mark.unit
    def test_remove_config_from_project(
        self, service: ProjectService, temp_dir: Path, sample_project_json: dict
    ):
        """Test removing a config from project."""
        project_file = temp_dir / "project.json"
        project_file.write_text(json.dumps(sample_project_json))

        project = service.remove_config_from_project(str(project_file), "config1")

        assert len(project.configs) == 0

    @pytest.mark.unit
    def test_remove_nonexistent_config(
        self, service: ProjectService, temp_dir: Path, sample_project_json: dict
    ):
        """Test removing nonexistent config (silent)."""
        project_file = temp_dir / "project.json"
        project_file.write_text(json.dumps(sample_project_json))

        project = service.remove_config_from_project(str(project_file), "nonexistent")

        assert len(project.configs) == 1


class TestProjectServiceListRecent:
    """Tests for listing recent projects."""

    @pytest.fixture
    def service(self) -> ProjectService:
        return ProjectService()

    @pytest.mark.unit
    def test_list_recent_projects(
        self, service: ProjectService, temp_dir: Path, sample_project_json: dict
    ):
        """Test listing recent valid projects."""
        project1 = temp_dir / "proj1" / "project.json"
        project1.parent.mkdir()
        sample_project_json["name"] = "Project 1"
        project1.write_text(json.dumps(sample_project_json))

        project2 = temp_dir / "proj2" / "project.json"
        project2.parent.mkdir()
        sample_project_json["name"] = "Project 2"
        project2.write_text(json.dumps(sample_project_json))

        paths = [str(project1), str(project2), "/nonexistent/project.json"]
        valid = service.list_recent_projects(paths)

        assert len(valid) == 2
        names = [name for _, name in valid]
        assert "Project 1" in names
        assert "Project 2" in names

    @pytest.mark.unit
    def test_list_recent_projects_all_invalid(self, service: ProjectService):
        """Test listing with all invalid paths."""
        paths = ["/invalid/1.json", "/invalid/2.json"]
        valid = service.list_recent_projects(paths)

        assert valid == []
