import itertools
import os
from typing import List

from colorama import Fore, Style

from pyftle.data_source import BatchSource, FileBatchSource
from pyftle.decorators import time_it
from pyftle.file_utils import get_files_list
from pyftle.file_writers import create_writer
from pyftle.ftle_solver import FTLESolver
from pyftle.hyperparameters import parse_args
from pyftle.integrate import create_integrator
from pyftle.interpolate import create_interpolator
from pyftle.parallel import ParallelExecutor


class MultipleFTLEProcessManager:
    """
    Manage the execution of multiple FTLE (Finite-Time Lyapunov Exponent)
    computations in parallel.

    This class automates the setup and execution of several FTLE solvers
    across multiple time windows (batches). It reads lists of velocity,
    coordinate, and particle files, organizes them into overlapping batches,
    and runs the solvers concurrently using a multiprocessing backend.

    The manager supports both forward- and backward-time FTLE computation,
    depending on the sign of the snapshot timestep.

    Attributes
    ----------
    snapshot_files : list[str]
        List of velocity field snapshot files.
    coordinate_files : list[str]
        List of coordinate grid files corresponding to each velocity snapshot.
    particle_files : list[str]
        List of files containing particle seed locations.
    timestep : float
        Time interval between consecutive velocity snapshots.
        A negative value indicates backward-time FTLE computation.
    executor : ParallelExecutor
        Object responsible for running FTLE solver instances in parallel.
    flow_grid_shape : tuple[int, int] | tuple[int, int, int]
        Shape of the flow field grid used by the interpolator.
    particles_grid_shape : tuple[int, int] | tuple[int, int, int]
        Shape of the particle seed grid.
    integrator : Integrator
        Object responsible for particle advection integration.
    writer : BaseWriter
        Output writer used to store FTLE results in the selected format.

    Notes
    -----
    This class is typically invoked via the main script entry point
    (e.g., `python -m pyftle.run_ftle`) and uses configuration parameters
    defined in `pyftle.hyperparameters.args`.
    """

    def __init__(self):
        """
        Initialize the FTLE process manager.

        Reads the file lists for velocity, coordinates, and particles;
        sets up interpolator, integrator, and output writer components;
        and determines whether computation will run forward or backward
        in time based on the sign of `args.snapshot_timestep`.
        """
        args = parse_args()

        self.snapshot_files: List[str] = get_files_list(args.list_velocity_files)
        self.coordinate_files: List[str] = get_files_list(args.list_coordinate_files)
        self.particle_files: List[str] = get_files_list(args.list_particle_files)
        self.timestep: float = args.snapshot_timestep
        self.flow_map_period = args.flow_map_period
        self.executor = ParallelExecutor(n_processes=args.num_processes)
        self.flow_grid_shape = args.flow_grid_shape
        self.particles_grid_shape = args.particles_grid_shape

        interpolator = create_interpolator(args.interpolator, self.flow_grid_shape)

        self.integrator = create_integrator(args.integrator, interpolator)

        output_dir = os.path.join("outputs", args.experiment_name)
        self.writer = create_writer(
            args.output_format, output_dir, args.particles_grid_shape
        )

        self._handle_time_direction()

    def _handle_time_direction(self) -> None:
        """
        Reverse file order for backward-time FTLE computation.

        If the timestep is negative, this method reverses the order of
        velocity, coordinate, and particle files to ensure consistent
        temporal progression during backward integration.

        Side Effects
        ------------
        Prints a message indicating the selected time direction.
        """
        if self.timestep < 0:
            self.snapshot_files.reverse()
            self.coordinate_files.reverse()
            self.particle_files.reverse()
            print("Running backward-time FTLE")
        else:
            print("Running forward-time FTLE")

    def _create_batches(self) -> list[BatchSource]:
        """
        Create overlapping batches of snapshot, coordinate, and particle files.

        Each batch corresponds to one flow-map integration period and contains
        the set of snapshots required to compute one FTLE field. The batches
        are constructed with cyclic repetition of coordinate and particle
        files to ensure proper coverage over time.

        Returns
        -------
        batches : list[BatchSource]
            List of batch objects containing file paths and metadata
            for each FTLE computation task.

        Notes
        -----
        The number of snapshots per batch is determined by:

            p = int(flow_map_period / abs(snapshot_timestep)) + 1

        resulting in `(n - p + 1)` overlapping batches for `n` snapshots total.
        """
        num_snapshots_total = len(self.snapshot_files)
        num_snapshots_in_flow_map_period = (
            int(self.flow_map_period / abs(self.timestep)) + 1
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
            )
            tasks.append(task)
        return tasks

    def _worker(self, batch_source: BatchSource, progress_queue):
        """
        Execute one FTLE computation task in a parallel process.

        Parameters
        ----------
        batch_source : BatchSource
            Object providing access to velocity, coordinate, and particle
            data for a single FTLE integration window.
        progress_queue : multiprocessing.Queue or None
            Shared queue used for progress tracking. Can be None when
            running in a single process (debug mode).
        """
        solver = FTLESolver(
            batch_source,
            integrator=self.integrator,
            progress_queue=progress_queue,
            output_writer=self.writer,
        )
        solver.run()

    def run(self):
        """
        Run all FTLE computations in parallel.

        This method creates batches of data using `_create_batches()` and
        dispatches each to the `ParallelExecutor`, which launches separate
        processes to execute `_worker()` concurrently.

        Notes
        -----
        If you want to debug the computation without multiprocessing,
        comment out the parallel call and uncomment the single-process
        line:

            # self._worker(batches[0], None)
        """
        batches = self._create_batches()
        self.executor.run(batches, self._worker)
        # self._worker(batches[0], None)  # For debugging purposes


# ─────────────────────────────────────────────────────────────
# Usage Example
# ─────────────────────────────────────────────────────────────


@time_it
def main():
    """
    Entry point for running the multiple FTLE process manager.

    Initializes the manager, creates the necessary components, and executes
    all FTLE computations in parallel. The total runtime is measured using
    the `@time_it` decorator.

    Raises
    ------
    RuntimeError
        If any FTLE solver fails during execution (e.g., due to invalid
        input data or inconsistent configuration).
    """
    try:
        manager = MultipleFTLEProcessManager()
        manager.run()
    except RuntimeError as e:
        print(f"{Fore.RED}\n❌ Execution stopped: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
