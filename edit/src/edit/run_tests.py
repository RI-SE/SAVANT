import pytest
import sys

def main():
    # Construct the arguments for pytest
    # We want to run tests in 'edit/tests' and generate coverage for the 'edit' module.
    args = [
        "edit/tests",
        "--cov=edit",
        "--cov-report=term-missing",
        "-v"  # verbose output
    ]
    # Execute pytest with the arguments
    # Note: uv run --package <package_name> will ensure that the package is in the Python path
    # and its dependencies are available.
    sys.exit(pytest.main(args))

if __name__ == "__main__":
    main()
