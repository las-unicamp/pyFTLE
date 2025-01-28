import glob
import os


def find_files_with_pattern(root_dir: str, pattern: str) -> list[str]:
    """
    Searches for files in the given root directory and its subdirectories that match
    the specified pattern.

    Args:
        root_dir (str): The root directory to start the search.
        pattern (str): The file extension or pattern to match (e.g., "*.txt", "*.csv").

    Returns:
        list[str]: A sorted list of full file paths that match the pattern.
    """
    # Ensure the root directory exists
    if not os.path.exists(root_dir):
        raise ValueError(f"The directory {root_dir} does not exist.")

    # Use glob to recursively find files matching the pattern
    search_pattern = os.path.join(root_dir, "**", pattern)
    matching_files = glob.glob(search_pattern, recursive=True)

    # Sort the list of files
    return sorted(matching_files)
