# markit Tests

Test suite for the markit video processing tool.

## Test Structure

```
tests/
├── conftest.py              # Shared pytest fixtures
├── test_geometry.py         # IoU calculations and polygon operations
├── test_config.py           # Configuration parsing and ontology loading
├── test_postprocessing.py   # Postprocessing passes
├── test_openlabel.py        # OpenLabel JSON generation and validation
├── test_integration.py      # End-to-end pipeline tests
└── fixtures/                # Test data
    ├── Kraklanda_short.mp4  # Test video file
    ├── best.pt              # Test YOLO model
    └── sample_outputs/      # Expected outputs for comparison
```

## Installation

Install test dependencies:

```bash
pip install -r requirements-dev.txt
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_geometry.py
```

### Run specific test class
```bash
pytest tests/test_geometry.py::TestBBoxOverlapCalculator
```

### Run specific test function
```bash
pytest tests/test_geometry.py::TestBBoxOverlapCalculator::test_identical_bboxes_return_iou_one
```

### Run with coverage report
```bash
pytest --cov=markitlib --cov-report=html
```

### Run only unit tests (fast)
```bash
pytest -m "not integration"
```

### Run only integration tests
```bash
pytest -m integration
```

### Run with verbose output
```bash
pytest -v
```

### Run with detailed output and print statements
```bash
pytest -vv -s
```

## Test Categories

### Unit Tests
- **test_geometry.py**: Tests for IoU calculations, polygon area, and bounding box operations
- **test_config.py**: Tests for configuration parsing, validation, and ontology loading
- **test_postprocessing.py**: Tests for postprocessing pipeline and individual passes
- **test_openlabel.py**: Tests for OpenLabel JSON structure generation and schema validation

### Integration Tests
- **test_integration.py**: End-to-end tests using real video files and YOLO models

## Test Data

The `fixtures/` directory contains:
- `Kraklanda_short.mp4`: Sample video file for testing video processing
- `best.pt`: Trained YOLO model weights for object detection
- `sample_outputs/`: Expected output files for comparison tests

## Writing New Tests

When adding new tests:

1. Follow the existing naming convention: `test_*.py`
2. Use fixtures from `conftest.py` for common test data
3. Add new fixtures to `conftest.py` if needed
4. Document test assumptions and expected behavior
5. Use descriptive test names that explain what is being tested

Example:
```python
def test_feature_does_specific_thing(self, sample_fixture):
    """Test that feature does X when Y happens."""
    # Arrange
    setup_data = prepare_test_data()

    # Act
    result = feature_under_test(setup_data)

    # Assert
    assert result == expected_value
```

## Continuous Integration

These tests can be integrated into CI/CD pipelines. Example GitHub Actions workflow:

```yaml
- name: Install dependencies
  run: pip install -r requirements-dev.txt

- name: Run tests
  run: pytest --cov=markitlib

- name: Upload coverage
  run: codecov
```

## Troubleshooting

### Missing test video or model files
The test fixtures must be present in `tests/fixtures/`. If missing:
- Video: `Kraklanda_short.mp4` should be a valid MP4 file
- Model: `best.pt` should be a valid YOLO model file

### YOLO model loading issues
Integration tests require the YOLO model to load properly. Ensure:
- Ultralytics is installed: `pip install ultralytics`
- Model file is compatible with installed ultralytics version

### Schema validation errors
If OpenLabel tests fail with schema validation errors:
- Ensure `savant_openlabel_subset.schema.json` exists in markit root
- Ensure `jsonschema` is installed: `pip install jsonschema`
