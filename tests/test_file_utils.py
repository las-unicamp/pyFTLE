import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd

from pyftle.file_utils import (
    find_files_with_pattern,
    get_files_list,
    write_list_to_txt,
)


class TestFileUtils(unittest.TestCase):
    @patch("pathlib.Path.rglob")
    def test_find_files_with_pattern(self, mock_rglob):
        """Tests that find_files_with_pattern correctly identifies files.

        Args:
            mock_rglob (MagicMock): Mock object for pathlib.Path.rglob.

        Flow:
            mock_rglob.return_value (list of Paths) -> find_files_with_pattern -> result (list of strings)
            result == expected list of strings
            mock_rglob.assert_called_with("*file*")
        """
        mock_rglob.return_value = [
            Path("root_dir/file1.txt"),
            Path("root_dir/file2.txt"),
        ]

        result = find_files_with_pattern("root_dir", "file")

        self.assertEqual(result, ["root_dir/file1.txt", "root_dir/file2.txt"])
        mock_rglob.assert_called_with("*file*")

    @patch("builtins.open", new_callable=mock_open)
    def test_write_list_to_txt(self, mock_file):
        """Tests that write_list_to_txt correctly writes a list to a file.

        Args:
            mock_file (MagicMock): Mock object for builtins.open.

        Flow:
            file_list, output_file -> write_list_to_txt
            mock_file.assert_called_once_with(output_file, "w")
            mock_file().write.assert_any_call for each item in file_list
        """
        file_list = ["file1.txt", "file2.txt"]
        output_file = "output.txt"

        write_list_to_txt(file_list, output_file)

        mock_file.assert_called_once_with(output_file, "w")
        mock_file().write.assert_any_call("file1.txt\n")
        mock_file().write.assert_any_call("file2.txt\n")

    @patch("os.path.exists")
    @patch("pandas.read_csv")
    def test_get_files_list_exists(self, mock_read_csv, mock_exists):
        """Tests that get_files_list correctly reads a file when it exists.

        Args:
            mock_read_csv (MagicMock): Mock object for pandas.read_csv.
            mock_exists (MagicMock): Mock object for os.path.exists.

        Flow:
            mock_exists.return_value = True
            mock_read_csv.return_value (DataFrame) -> get_files_list -> result (list)
            result == expected list
            mock_exists.assert_called_once_with
            mock_read_csv.assert_called_once_with
        """
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame(
            {0: ["file1.txt", "file2.txt", "file3.txt"]}
        )

        result = get_files_list("velocity_file.csv")

        self.assertEqual(result, ["file1.txt", "file2.txt", "file3.txt"])
        mock_exists.assert_called_once_with("velocity_file.csv")
        mock_read_csv.assert_called_once_with(
            "velocity_file.csv", header=None, dtype=str
        )

    @patch("os.path.exists")
    def test_get_files_list_not_exists(self, mock_exists):
        """Tests that get_files_list raises FileNotFoundError when the file does not exist.

        Args:
            mock_exists (MagicMock): Mock object for os.path.exists.

        Flow:
            mock_exists.return_value = False -> get_files_list -> raises FileNotFoundError
        """
        mock_exists.return_value = False

        with self.assertRaises(FileNotFoundError):
            get_files_list("non_existent_file.csv")


if __name__ == "__main__":
    unittest.main()