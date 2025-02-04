import itertools
from typing import List

from tqdm import tqdm

from src.file_readers import (
    read_coordinates,
    read_seed_particles_coordinates,
    read_velocity_data,
)
from src.file_utils import get_files_list
from src.hyperparameters import args
from src.integrate import EulerIntegrator
from src.interpolate import CubicInterpolatorStrategy
from src.particle_position import PositionDict


def validate_input_lists(velocity_list: List[str], grid_list: List[str]) -> None:
    if len(grid_list) > 1:  # if not a single grid file
        assert len(velocity_list) == len(grid_list)


def main():
    # Load the lists for the velocity and grid files
    snapshot_files = get_files_list(args.list_velocity_files)
    grid_files = get_files_list(args.list_grid_files)
    particle_files = get_files_list(args.list_particle_files)

    validate_input_lists(snapshot_files, grid_files)

    num_snapshots_in_flow_map_period = (
        int(args.flow_map_period / args.snapshot_timestep) + 1
    )

    progress_bar = tqdm(
        total=num_snapshots_in_flow_map_period,
        leave=True,
        desc="Progress",
    )

    start_index = 0  # Example: Start from index 2

    snapshot_files_period = snapshot_files[
        start_index : start_index + num_snapshots_in_flow_map_period
    ]
    # Repeat the first grid file if there's only one, otherwise slice normally
    grid_files_period = list(
        itertools.islice(
            itertools.cycle(grid_files),
            start_index,
            start_index + num_snapshots_in_flow_map_period,
        )
    )
    # Repeat the first seed file if there's only one, otherwise slice normally
    particles_files = list(
        itertools.islice(itertools.cycle(particle_files), len(snapshot_files))
    )

    particle_file = particles_files[start_index]

    current_position = PositionDict(read_seed_particles_coordinates(particle_file))
    integrator = EulerIntegrator()

    for snapshot_file, grid_file in zip(snapshot_files_period, grid_files_period):
        tqdm.write(f"Snapshot: {snapshot_file}, Grid: {grid_file}")

        velocities = read_velocity_data(snapshot_file)
        coordinates = read_coordinates(grid_file)

        interpolator = CubicInterpolatorStrategy(
            coordinates, velocities[:, 0], velocities[:, 1]
        )

        current_position = integrator.integrate(
            args.snapshot_timestep, current_position, interpolator
        )

        progress_bar.update(1)

    progress_bar.close()

    # for i, (snapshot, grid) in enumerate(
    #     itertools.zip_longest(
    #         snapshot_files, grid_files * len(snapshot_files), fillvalue=grid_files[0]
    #     )
    # ):
    #     particles = read_seed_particles_coordinates(args.particles_filename)
    #     print(f"Snapshot: {snapshot}, Grid: {grid}")

    #     progress_bar.update(1)


if __name__ == "__main__":
    main()
