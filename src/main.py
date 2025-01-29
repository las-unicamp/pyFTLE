import itertools
import time
from typing import List

from tqdm import tqdm

from src.file_utils import get_grid_files_list, get_velocity_files_list
from src.hyperparameters import args


def validate_input_lists(velocity_list: List[str], grid_list: List[str]) -> None:
    if len(grid_list) > 1:  # if not a single grid file
        assert len(velocity_list) == len(grid_list)


def main():
    # Load the lists for the velocity and grid files
    snapshot_files = get_velocity_files_list(args.list_velocity_files)
    grid_files = get_grid_files_list(args.list_grid_files)

    validate_input_lists(snapshot_files, grid_files)

    progress_bar = tqdm(
        total=len(snapshot_files),
        leave=True,
        desc="Progress",
    )

    # Iterate over both velocity and grid lists, also handling single grid cases
    for snapshot, grid in itertools.zip_longest(
        snapshot_files, grid_files * len(snapshot_files), fillvalue=grid_files[0]
    ):
        print(f"Snapshot: {snapshot}, Grid: {grid}")
        time.sleep(0.02)

        progress_bar.update(1)


if __name__ == "__main__":
    main()
