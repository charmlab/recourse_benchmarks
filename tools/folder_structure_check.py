import os
import sys

import pytest

EXEMPTED_METHODS = [
    "__pycache__",
    "actionable_recourse",
    "causal_recourse",
    "cchvae",
    "cem",
    "clue",
    "crud",
    "dice",
    "face",
    "feature_tweak",
    "focus",
    "growing_spheres",
    "mace",
    "revise",
    "roar",
    "wachter",
    "claproar",
    "gravitational",
    "greedy",
]
REQUIRED_FILES = ["__init__.py", "model.py", "reproduce.py"]
ROOT_FOLDER = os.path.join(
    os.path.dirname(__file__), "..", "recourse_methods", "catalog"
)


def check_folder_structure(folder_path):
    """Checks if the folder contains the required files."""
    # Get the list of files and directories inside the folder
    found_files = set(os.listdir(folder_path))
    print(f"Found files: {found_files}")

    # Check for the required files in each subfolder
    missing_files = [file for file in REQUIRED_FILES if file not in found_files]
    if missing_files:
        print(
            f"Error: Missing required files in {folder_path}: {', '.join(missing_files)}"
        )
        return False

    # Check if test.py exists and run unit tests
    test_file_path = os.path.join(folder_path, "reproduce.py")
    if os.path.exists(test_file_path):
        # # Change directory to the folder containing test.py to avoid import issues
        # original_dir = os.getcwd()
        # os.chdir(folder_path)
        # Run pytest on the test.py file and check if it passes
        exit_code = pytest.main([test_file_path])
        # result = subprocess.run(["pytest", test_file_path], capture_output=True)
        # os.chdir(original_dir)  # Return to the original directory

        if exit_code != 0:
            print(f"Error: Unit tests failed in {test_file_path}")
            return False
    else:
        print(f"Error: test.py not found in {folder_path}")
        return False

    return True


def check_all_folders_in_new_methods():
    """Check all subfolders in the new_methods directory."""
    root_path = os.path.join(os.getcwd(), ROOT_FOLDER)

    # Check if the root folder exists
    if not os.path.exists(root_path):
        print(f"Error: {ROOT_FOLDER} folder not found.")
        return False

    # Walk through all subfolders inside new_methods
    all_passed = True
    for folder_name in os.listdir(root_path):
        if folder_name in EXEMPTED_METHODS:
            continue
        folder_path = os.path.join(root_path, folder_name)

        # Only check directories
        if os.path.isdir(folder_path):
            print(f"Checking folder: {folder_name}")
            passed = check_folder_structure(folder_path)
            if not passed:
                all_passed = False

    return all_passed


if __name__ == "__main__":
    # Check all folders and return appropriate exit code
    if not check_all_folders_in_new_methods():
        sys.exit(1)  # Exit with failure if checks failed
    sys.exit(0)  # Exit successfully if all checks passed
