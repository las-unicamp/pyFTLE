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


class SnapshotProcessor:
    """Handles the computation of FTLE for a single snapshot period."""

    def __init__(
        self,
        index: int,
        snapshot_files: List[str],
        grid_files: List[str],
        particle_file: str,
        tqdm_position_queue,
        progress_dict,
    ):
        self.index = index
        self.snapshot_files = snapshot_files
        self.grid_files = grid_files
        self.particle_file = particle_file
        self.progress_dict = progress_dict
        self.tqdm_position_queue = tqdm_position_queue
        self.tqdm_position = None  # Will be assigned dynamically
        self.output_dir = f"outputs/{args.experiment_name}"

    def run(self):
        """Processes a single snapshot period."""
        self.tqdm_position = self.tqdm_position_queue.get()

        # Force clean ghost tqdm_bar bars before starting a new one
        if hasattr(self, "tqdm_bar"):
            self.tqdm_bar.clear()
            self.tqdm_bar.close()

        tqdm_bar = tqdm(
            total=len(self.snapshot_files),
            desc=f"FTLE {self.index:04d}",
            position=self.tqdm_position,
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
        )

        particles = read_seed_particles_coordinates(self.particle_file)
        integrator = get_integrator(args.integrator)
        velocity_reader = VelocityDataReader()
        coordinate_reader = CoordinateDataReader()
        interpolator_factory = InterpolatorFactory(coordinate_reader, velocity_reader)

        for snapshot_file, grid_file in zip(self.snapshot_files, self.grid_files):
            tqdm_bar.set_description(f"FTLE {self.index:04d}: {snapshot_file}")
            tqdm_bar.update(1)

            interpolator = interpolator_factory.create_interpolator(
                snapshot_file, grid_file, args.interpolator
            )
            integrator.integrate(args.snapshot_timestep, particles, interpolator)

        self._compute_and_save_ftle(particles)

        tqdm_bar.clear()
        tqdm_bar.close()
        self.progress_dict[self.index] = True  # Notify progress monitor
        self.tqdm_position_queue.put(self.tqdm_position)

    def _compute_and_save_ftle(self, particles):
        """Computes FTLE and saves the results."""
        jacobian = compute_flow_map_jacobian(particles)
        map_period = (len(self.snapshot_files) - 1) * abs(args.snapshot_timestep)
        ftle_field = compute_ftle(jacobian, map_period)

        os.makedirs(self.output_dir, exist_ok=True)

        filename = os.path.join(self.output_dir, f"ftle{self.index:04d}.mat")
        savemat(filename, {"ftle": ftle_field})


class FTLEComputationManager:
    """Manages the distribution of snapshot processing tasks."""

    def __init__(self):
        self.snapshot_files = get_files_list(args.list_velocity_files)
        self.grid_files = get_files_list(args.list_grid_files)
        self.particle_files = get_files_list(args.list_particle_files)
        self._validate_input_lists()

        self.num_snapshots_total = len(self.snapshot_files)
        self.num_snapshots_in_flow_map_period = (
            int(args.flow_map_period / abs(args.snapshot_timestep)) + 1
        )
        self.num_processes = args.num_processes

        self._handle_time_direction()

    def _validate_input_lists(self):
        """Ensures input lists are correctly formatted."""
        if len(self.grid_files) > 1:
            assert len(self.snapshot_files) == len(self.grid_files)
        if len(self.particle_files) > 1:
            assert len(self.snapshot_files) == len(self.particle_files)

    def _handle_time_direction(self):
        """Handles time direction for backward/forward FTLE computation."""
        if args.snapshot_timestep < 0:
            self.snapshot_files.reverse()
            self.grid_files.reverse()
            self.particle_files.reverse()
            print("Running backward-time FTLE")
        else:
            print("Running forward-time FTLE")

    def run(self):
        """Runs FTLE computation using multiprocessing with shared progress tracking."""
        pool = multiprocessing.Pool(processes=self.num_processes)
        manager = multiprocessing.Manager()
        progress_dict = manager.dict()
        tqdm_position_queue = manager.Queue()

        # Initialize available tqdm positions (from 1 to num_processes)
        for i in range(1, self.num_processes + 1):
            tqdm_position_queue.put(i)

        tqdm_outer = tqdm(
            total=self.num_snapshots_total - self.num_snapshots_in_flow_map_period + 1,
            desc="Total Progress",
            position=0,
            leave=True,
        )

        tasks = []
        for i in range(
            self.num_snapshots_total - self.num_snapshots_in_flow_map_period + 1
        ):
            snapshot_files_period = self.snapshot_files[
                i : i + self.num_snapshots_in_flow_map_period
            ]
            grid_files_period = list(
                itertools.islice(
                    itertools.cycle(self.grid_files),
                    i,
                    i + self.num_snapshots_in_flow_map_period,
                )
            )
            particle_file = list(
                itertools.islice(
                    itertools.cycle(self.particle_files), self.num_snapshots_total
                )
            )[i]

            progress_dict[i] = False  # Mark as incomplete

            processor = SnapshotProcessor(
                i,
                snapshot_files_period,
                grid_files_period,
                particle_file,
                tqdm_position_queue,
                progress_dict,
            )
            tasks.append(pool.apply_async(processor.run))

        self._monitor_progress(tasks, progress_dict, tqdm_outer)

        pool.close()
        pool.join()
        tqdm_outer.close()

    def _monitor_progress(self, tasks, tqdm_dict, tqdm_outer):
        """Monitors the completion of parallel tasks and updates the progress bar."""
        completed = 0
        while completed < len(tasks):
            completed = sum(1 for v in tqdm_dict.values() if v)  # Count completed tasks
            tqdm_outer.update(completed - tqdm_outer.n)  # Increment new completions
            tqdm_outer.refresh()
            time.sleep(2.0)  # Prevents excessive polling, keeping CPU usage low

            # OBS: the computations inside the loop runs asynchronously, thus
            # time.sleep will not held the computations. The present method just
            # waits for notifications of completed tasks.


@timeit
def main():
    """Main execution entry point."""
    manager = FTLEComputationManager()
    manager.run()


if __name__ == "__main__":
    main()
