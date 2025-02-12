import itertools
import os
from typing import List

from scipy.io import savemat
from tqdm import tqdm

from src.cauchy_green import compute_flow_map_jacobian
from src.decorators import timeit
from src.file_readers import (
    read_seed_particles_coordinates,
)
from src.file_utils import get_files_list
from src.ftle import compute_ftle
from src.hyperparameters import args
from src.integrate import get_integrator
from src.interpolate import create_interpolator


def validate_input_lists(
    velocity_list: List[str], grid_list: List[str], particle_list: List[str]
) -> None:
    if len(grid_list) > 1:  # if not a single grid file
        assert len(velocity_list) == len(grid_list)
    if len(particle_list) > 1:  # if not a single particle file
        assert len(velocity_list) == len(particle_list)


@timeit
def main():
    # Load the lists for the velocity and grid files
    snapshot_files = get_files_list(args.list_velocity_files)
    grid_files = get_files_list(args.list_grid_files)
    particle_files = get_files_list(args.list_particle_files)

    validate_input_lists(snapshot_files, grid_files, particle_files)

    num_snapshots_total = len(snapshot_files)
    num_snapshots_in_flow_map_period = (
        int(args.flow_map_period / abs(args.snapshot_timestep)) + 1
    )

    is_backward = args.snapshot_timestep < 0
    if is_backward:
        snapshot_files.reverse()
        grid_files.reverse()
        particle_files.reverse()
        print("Running backward-time FTLE")
    else:
        print("Running forward-time FTLE")

    for i in tqdm(
        range(num_snapshots_total - num_snapshots_in_flow_map_period + 1),
        desc="Total Progress",
        leave=True,
    ):
        snapshot_files_period = snapshot_files[i : i + num_snapshots_in_flow_map_period]
        # Repeat the first grid file if there's only one, otherwise slice normally
        grid_files_period = list(
            itertools.islice(
                itertools.cycle(grid_files),
                i,
                i + num_snapshots_in_flow_map_period,
            )
        )
        # Repeat the first seed file if there's only one, otherwise slice normally
        particles_files = list(
            itertools.islice(itertools.cycle(particle_files), num_snapshots_total)
        )

        particle_file = particles_files[i]

        particles = read_seed_particles_coordinates(particle_file)
        integrator = get_integrator(args.integrator)

        for snapshot_file, grid_file in tqdm(
            zip(snapshot_files_period, grid_files_period),
            desc="Single FTLE Progress",
            leave=True,
            total=num_snapshots_in_flow_map_period,
        ):
            tqdm.write(f"Snapshot: {snapshot_file}, Grid: {grid_file}")

            interpolator = create_interpolator(snapshot_file, grid_file)

            integrator.integrate(args.snapshot_timestep, particles, interpolator)

        jacobian = compute_flow_map_jacobian(particles)
        map_period = (num_snapshots_in_flow_map_period - 1) * abs(
            args.snapshot_timestep
        )
        ftle_field = compute_ftle(jacobian, map_period)

        output_dir = "outputs/vawt_naca0018"
        filename = os.path.join(output_dir, f"ftle{i:04d}.mat")
        savemat(
            filename,
            {
                "ftle": ftle_field,
            },
        )


if __name__ == "__main__":
    main()
