import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd

from src.file_utils import (
    find_files_with_pattern,
    get_files_list,
    write_list_to_txt,
)


class TestFileUtils(unittest.TestCase):
    @patch("pathlib.Path.rglob")
    def test_find_files_with_pattern(self, mock_rglob):
        # Setup mock
        mock_rglob.return_value = [
            Path("root_dir/file1.txt"),
            Path("root_dir/file2.txt"),
        ]

        # Test
        result = find_files_with_pattern("root_dir", "file")

        # Verify
        self.assertEqual(result, ["root_dir/file1.txt", "root_dir/file2.txt"])
        mock_rglob.assert_called_with("*file*")

    @patch("builtins.open", new_callable=mock_open)
    def test_write_list_to_txt(self, mock_file):
        # Test data
        file_list = ["file1.txt", "file2.txt"]
        output_file = "output.txt"

        # Call the function
        write_list_to_txt(file_list, output_file)

        # Verify file write
        mock_file.assert_called_once_with(output_file, "w")
        mock_file().write.assert_any_call("file1.txt\n")
        mock_file().write.assert_any_call("file2.txt\n")

    @patch("os.path.exists")
    @patch("pandas.read_csv")
    def test_get_files_list_exists(self, mock_read_csv, mock_exists):
        # Mock
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame(
            {0: ["file1.txt", "file2.txt", "file3.txt"]}
        )  # Mocking data without headers

        # Test
        result = get_files_list("velocity_file.csv")

        # Verify
        self.assertEqual(result, ["file1.txt", "file2.txt", "file3.txt"])  # Flat list
        mock_exists.assert_called_once_with("velocity_file.csv")
        mock_read_csv.assert_called_once_with("velocity_file.csv", header=None)

    @patch("os.path.exists")
    def test_get_files_list_not_exists(self, mock_exists):
        # Mock
        mock_exists.return_value = False

        # Test and verify exception
        with self.assertRaises(FileNotFoundError):
            get_files_list("non_existent_file.csv")


if __name__ == "__main__":
    unittest.main()
