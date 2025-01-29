import os
from pathlib import Path

import pandas as pd


def find_files_with_pattern(root_dir: str, pattern: str) -> list[str]:
    """
    Searches for files in the given root directory and its subdirectories that match
    the specified pattern.

    Args:
        root_dir (str): The root directory to start the search.
        pattern (str): The pattern to match in filenames.

    Returns:
        list[str]: A sorted list of full file paths that match the pattern.
    """
    root_path = Path(root_dir)

    # Search recursively for files containing the pattern
    matching_files = sorted(str(file) for file in root_path.rglob(f"*{pattern}*"))

    return matching_files


def write_list_to_txt(file_list: list[str], output_file: str):
    """
    Writes a list of file paths to a text file.

    Args:
        file_list (list[str]): List of file paths.
        output_file (str): Output text file path.
    """
    with open(output_file, "w") as f:
        for file_path in file_list:
            f.write(file_path + "\n")


def get_velocity_files_list(file_path: str) -> pd.DataFrame:
    """
    Reads a velocity file and returns its content as a pandas DataFrame.

    Args:
        file_path (str): Path to the velocity file.

    Returns:
        pd.DataFrame: DataFrame containing the velocity data.
    """
    if os.path.exists(file_path):
        velocity_data = pd.read_csv(file_path, header=None)
        return velocity_data.iloc[:, 0].tolist()
    else:
        raise FileNotFoundError(f"Velocity file not found at {file_path}")


def get_grid_files_list(file_path: str) -> pd.DataFrame:
    """
    Reads a grid file and returns its content as a pandas DataFrame.

    Args:
        file_path (str): Path to the grid file.

    Returns:
        pd.DataFrame: DataFrame containing the grid data.
    """
    if os.path.exists(file_path):
        grid_data = pd.read_csv(file_path, header=None)
        return grid_data.iloc[:, 0].tolist()
    else:
        raise FileNotFoundError(f"Grid file not found at {file_path}")
