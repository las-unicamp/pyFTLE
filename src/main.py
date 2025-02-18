import itertools
import multiprocessing
import os
import time
from typing import List

from scipy.io import savemat
from tqdm import tqdm

from src.cauchy_green import compute_flow_map_jacobian
from src.decorators import timeit
from src.file_readers import (
    CoordinateDataReader,
    VelocityDataReader,
    read_seed_particles_coordinates,
)
from src.file_utils import get_files_list
from src.ftle import compute_ftle
from src.hyperparameters import args
from src.integrate import get_integrator
from src.interpolate import InterpolatorFactory


def validate_input_lists(
    velocity_list: List[str], grid_list: List[str], particle_list: List[str]
) -> None:
    if len(grid_list) > 1:  # if not a single grid file
        assert len(velocity_list) == len(grid_list)
    if len(particle_list) > 1:  # if not a single particle file
        assert len(velocity_list) == len(particle_list)


def process_single_snapshot(
    i, snapshot_files_period, grid_files_period, particle_file, tqdm_position, queue
):
    """Function to process a single snapshot in a separate process"""
    tqdm_bar = tqdm(
        total=len(snapshot_files_period),
        desc=f"FTLE {i:04d}",
        position=tqdm_position,
        leave=False,  # Do not leave it in terminal after completion
    )

    particles = read_seed_particles_coordinates(particle_file)
    integrator = get_integrator(args.integrator)
    velocity_reader = VelocityDataReader()
    coordinate_reader = CoordinateDataReader()
    interpolator_factory = InterpolatorFactory(coordinate_reader, velocity_reader)

    for snapshot_file, grid_file in zip(snapshot_files_period, grid_files_period):
        tqdm_bar.set_description(f"FTLE {i:04d}: {snapshot_file}")
        tqdm_bar.update()
        time.sleep(0.5)  # Simulating processing time

        interpolator = interpolator_factory.create_interpolator(
            snapshot_file, grid_file, args.interpolator
        )
        integrator.integrate(args.snapshot_timestep, particles, interpolator)

    jacobian = compute_flow_map_jacobian(particles)
    map_period = (len(snapshot_files_period) - 1) * abs(args.snapshot_timestep)
    ftle_field = compute_ftle(jacobian, map_period)

    output_dir = "outputs/vawt_naca0018"
    filename = os.path.join(output_dir, f"ftle{i:04d}.mat")
    savemat(filename, {"ftle": ftle_field})

    tqdm_bar.close()

    # Notify the main process that this task is done
    queue.put(1)


@timeit
def main():
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

    num_processes = 4  # Adjust the number of processes
    pool = multiprocessing.Pool(processes=num_processes)
    manager = multiprocessing.Manager()
    queue = manager.Queue()  # Shared queue to track completed jobs

    tqdm_outer = tqdm(
        total=num_snapshots_total - num_snapshots_in_flow_map_period + 1,
        desc="Total Progress",
        position=0,
        leave=True,
    )

    tasks = []
    for i in range(num_snapshots_total - num_snapshots_in_flow_map_period + 1):
        snapshot_files_period = snapshot_files[i : i + num_snapshots_in_flow_map_period]
        grid_files_period = list(
            itertools.islice(
                itertools.cycle(grid_files), i, i + num_snapshots_in_flow_map_period
            )
        )
        particles_files = list(
            itertools.islice(itertools.cycle(particle_files), num_snapshots_total)
        )
        particle_file = particles_files[i]

        # Assign each process a unique position for progress bars (1 to 4)
        tqdm_position = (i % num_processes) + 1

        tasks.append(
            pool.apply_async(
                process_single_snapshot,
                args=(
                    i,
                    snapshot_files_period,
                    grid_files_period,
                    particle_file,
                    tqdm_position,
                    queue,
                ),
            )
        )

    # Monitor completion in the main process
    completed = 0
    while completed < len(tasks):
        queue.get()  # Wait for a task to finish
        tqdm_outer.update()
        completed += 1

    pool.close()
    pool.join()
    tqdm_outer.close()


if __name__ == "__main__":
    main()
