import os
from tempfile import TemporaryDirectory

import pytest

from src.file_utils import find_files_with_pattern


def create_test_files(root_dir, files):
    """
    Helper function to create test files in the specified directory.
    Args:
        root_dir (str): The root directory for the files.
        files (list[str]): List of relative file paths to create.
    """
    for file in files:
        file_path = os.path.join(root_dir, file)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write("Test content")  # Add some test content to the file


def test_find_files_with_pattern_valid_directory():
    # Create a temporary directory with test files
    with TemporaryDirectory() as temp_dir:
        # Define test files
        test_files = [
            "file1.txt",
            "subdir/file2.txt",
            "subdir/file3.csv",
            "file4.txt",
            "file5.md",
        ]
        create_test_files(temp_dir, test_files)

        # Test for .txt files
        result = find_files_with_pattern(temp_dir, "*.txt")
        expected = sorted(
            [
                os.path.join(temp_dir, "file1.txt"),
                os.path.join(temp_dir, "subdir/file2.txt"),
                os.path.join(temp_dir, "file4.txt"),
            ]
        )
        assert result == expected

        # Test for .csv files
        result = find_files_with_pattern(temp_dir, "*.csv")
        expected = [os.path.join(temp_dir, "subdir/file3.csv")]
        assert result == expected

        # Test for .md files
        result = find_files_with_pattern(temp_dir, "*.md")
        expected = [os.path.join(temp_dir, "file5.md")]
        assert result == expected


def test_find_files_with_pattern_empty_directory():
    # Create an empty temporary directory
    with TemporaryDirectory() as temp_dir:
        result = find_files_with_pattern(temp_dir, "*.txt")
        assert result == []  # Expect an empty list


def test_find_files_with_pattern_nonexistent_directory():
    # Test with a non-existent directory
    with pytest.raises(ValueError, match="does not exist"):
        find_files_with_pattern("/nonexistent/directory", "*.txt")


def test_find_files_with_pattern_no_matching_files():
    # Create a temporary directory with files that do not match the pattern
    with TemporaryDirectory() as temp_dir:
        test_files = ["file1.md", "file2.csv"]
        create_test_files(temp_dir, test_files)

        result = find_files_with_pattern(temp_dir, "*.txt")
        assert result == []  # Expect an empty list
