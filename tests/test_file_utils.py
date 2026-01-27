from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from pyftle.file_utils import (
    find_files_with_pattern,
    get_files_list,
    write_list_to_txt,
)


@patch("pathlib.Path.rglob")
def test_find_files_with_pattern(mock_rglob):
    mock_rglob.return_value = [
        Path("root_dir/file1.txt"),
        Path("root_dir/file2.txt"),
    ]

    result = find_files_with_pattern("root_dir", "file")

    assert result == ["root_dir/file1.txt", "root_dir/file2.txt"]
    mock_rglob.assert_called_with("*file*")


@patch("builtins.open", new_callable=mock_open)
def test_write_list_to_txt(mock_file):
    file_list = ["file1.txt", "file2.txt"]
    output_file = "output.txt"

    write_list_to_txt(file_list, output_file)

    mock_file.assert_called_once_with(output_file, "w")
    mock_file().write.assert_any_call("file1.txt\n")
    mock_file().write.assert_any_call("file2.txt\n")


@patch("os.path.exists")
@patch("pandas.read_csv")
def test_get_files_list_exists(mock_read_csv, mock_exists):
    mock_exists.return_value = True
    mock_read_csv.return_value = pd.DataFrame(
        {0: ["file1.txt", "file2.txt", "file3.txt"]}
    )

    result = get_files_list("velocity_file.csv")

    assert result == ["file1.txt", "file2.txt", "file3.txt"]
    mock_exists.assert_called_once_with("velocity_file.csv")
    mock_read_csv.assert_called_once_with("velocity_file.csv", header=None, dtype=str)


@patch("os.path.exists")
def test_get_files_list_not_exists(mock_exists):
    mock_exists.return_value = False

    with pytest.raises(FileNotFoundError):
        get_files_list("non_existent_file.csv")
