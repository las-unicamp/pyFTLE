from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.file_utils import find_files_with_pattern, write_list_to_txt


# Test for find_files_with_pattern
@pytest.fixture
def mock_rglob():
    with patch("pathlib.Path.rglob") as mock:
        yield mock


def test_find_files_with_pattern(mock_rglob):
    # Arrange
    mock_rglob.return_value = [
        Path("/mock/path/file1.txt"),
        Path("/mock/path/file2.txt"),
    ]
    root_dir = "/mock/path"
    pattern = ".txt"

    # Act
    result = find_files_with_pattern(root_dir, pattern)

    # Assert
    assert result == ["/mock/path/file1.txt", "/mock/path/file2.txt"]
    mock_rglob.assert_called_once_with("*" + pattern + "*")


# Test for write_list_to_txt
@pytest.fixture
def mock_open():
    with patch("builtins.open", new_callable=MagicMock) as mock:
        yield mock


def test_write_list_to_txt(mock_open):
    # Arrange
    file_list = ["/mock/path/file1.txt", "/mock/path/file2.txt"]
    output_file = "/mock/output.txt"

    # Act
    write_list_to_txt(file_list, output_file)

    # Assert
    mock_open.assert_called_once_with(output_file, "w")
    mock_open.return_value.write.assert_any_call("/mock/path/file1.txt\n")
    mock_open.return_value.write.assert_any_call("/mock/path/file2.txt\n")
