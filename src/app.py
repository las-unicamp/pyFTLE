import itertools
import os
from typing import List

from colorama import Fore, Style

from src.data_source import BatchSource, FileBatchSource
from src.decorators import timeit
from src.file_utils import get_files_list
from src.file_writers import create_writer
from src.ftle_solver import FTLESolver
from src.hyperparameters import args
from src.integrate import create_integrator
from src.interpolate import create_interpolator
from src.parallel import ParallelExecutor


class MultipleFTLEProcessManager:
    """Reads snapshot list, creates batches, and runs FTLE solvers in parallel."""

    def __init__(self):
        self.snapshot_files: List[str] = get_files_list(args.list_velocity_files)
        self.coordinate_files: List[str] = get_files_list(args.list_coordinate_files)
        self.particle_files: List[str] = get_files_list(args.list_particle_files)
        self.timestep: float = args.snapshot_timestep
        self.executor = ParallelExecutor(n_processes=args.num_processes)
        self.grid_shape = args.grid_shape

        interpolator = create_interpolator(args.interpolator)

        self.integrator = create_integrator(args.integrator, interpolator)

        output_dir = os.path.join("outputs", args.experiment_name)
        self.writer = create_writer(args.output_format, output_dir, args.grid_shape)

        self._handle_time_direction()

    def _handle_time_direction(self) -> None:
        """Handles time direction for backward/forward FTLE computation."""
        if self.timestep < 0:
            self.snapshot_files.reverse()
            self.coordinate_files.reverse()
            self.particle_files.reverse()
            print("Running backward-time FTLE")
        else:
            print("Running forward-time FTLE")

    def _create_batches(self) -> list[BatchSource]:
        """Generate overlapping batches of snapshot, coordinate and particle files"""
        num_snapshots_total = len(self.snapshot_files)
        num_snapshots_in_flow_map_period = (
            int(args.flow_map_period / abs(args.snapshot_timestep)) + 1
        )

        p = num_snapshots_in_flow_map_period
        n = num_snapshots_total

        # Precompute snapshot file batches
        snapshot_batches = [self.snapshot_files[i : i + p] for i in range(n - p + 1)]

        # Precompute coordinate file batches (cycled)
        coord_cycle = list(
            itertools.islice(itertools.cycle(self.coordinate_files), n + p - 1)
        )
        coordinate_batches = [coord_cycle[i : i + p] for i in range(n - p + 1)]

        # Precompute particle file selection (cycled)
        particle_cycle = list(itertools.islice(itertools.cycle(self.particle_files), n))
        particle_batches = [particle_cycle[i] for i in range(n - p + 1)]  # pick one str

        tasks: list[BatchSource] = []

        for i in range(n - p + 1):
            task = FileBatchSource(
                snapshot_files=list(snapshot_batches[i]),
                coordinate_files=list(coordinate_batches[i]),
                particle_file=particle_batches[i],  # Assume single particle file
                snapshot_timestep=self.timestep,
                flow_map_period=p,
                grid_shape=self.grid_shape,
            )
            tasks.append(task)
        return tasks

    def _worker(self, batch_source: BatchSource, progress_queue):
        """Wrapper function for parallel execution."""
        solver = FTLESolver(
            batch_source,
            integrator=self.integrator,
            progress_queue=progress_queue,
            output_writer=self.writer,
        )
        solver.run()

    def run(self):
        batches = self._create_batches()
        self.executor.run(batches, self._worker)


# ─────────────────────────────────────────────────────────────
# Usage Example
# ─────────────────────────────────────────────────────────────


@timeit
def main():
    try:
        manager = MultipleFTLEProcessManager()
        manager.run()
    except RuntimeError as e:
        print(f"{Fore.RED}\n❌ Execution stopped: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
