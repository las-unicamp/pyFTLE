from datetime import datetime
from typing import Callable, Optional, cast

import numpy as np

from pyftle.data_source import AnalyticalBatchSource, BatchSource
from pyftle.file_writers import create_writer
from pyftle.ftle_solver import FTLESolver
from pyftle.integrate import create_integrator
from pyftle.interpolate import create_interpolator
from pyftle.parallel import ParallelExecutor
from pyftle.particles import NeighboringParticles


class AnalyticalSolver:
    """
    A notebook-friendly FTLE manager for in-memory analytical velocity fields.

    This class provides a high-level interface to compute Finite-Time Lyapunov
    Exponents (FTLE) from an analytical velocity field. It generates time batches,
    integrates particle trajectories, and computes flow maps entirely in memory,
    with optional parallelization and output writing.

    The solver is particularly suited for analytical or synthetic flow fields
    defined as Python functions rather than on a discrete spatial grid.

    Parameters
    ----------
    velocity_fn : Callable[[np.ndarray, float], np.ndarray]
        A user-defined function that returns the velocity field at a given time.
        The callable must have the signature:

        ``velocity_fn(positions: ArrayNx2 | ArrayNx3, t: float) -> ArrayNx2 | ArrayNx3``

        where ``positions`` are the spatial coordinates of the particles and
        ``t`` is the time.
    particles : NeighboringParticles
        Initial particle positions and neighbor information used for computing
        local flow map Jacobians.
    timestep : float
        Time step size used for particle integration. Positive values correspond
        to forward-time integration, and negative values to backward-time.
    flow_map_period : float
        Total duration over which each flow map is integrated (in the same units
        as `timestep`).
    num_ftles : int
        Number of FTLE fields to compute. Each FTLE field corresponds to a
        distinct start time separated by `timestep`.
    integrator_name : str
        Name of the time integrator to use. Must be one of the registered
        integrators available in :mod:`pyftle.integrate` (e.g., "rk4", "euler").
    num_processes : int, default=1
        Number of parallel processes used for computing FTLE fields.
    save_output : bool, default=False
        If True, the computed FTLE fields will be written to disk instead of
        being returned as NumPy arrays.
    output_format : str, default="vtk"
        File format used for writing the output (e.g., "vtk" or "npy").
    output_dir_name : Optional[str], default=None
        Output directory name. If not provided and `save_output=True`,
        a timestamped directory name (e.g., "run-2025-11-07-18h-42m-00s")
        will be automatically created.

    Attributes
    ----------
    velocity_fn : Callable
        Analytical velocity function used for interpolation.
    particles : NeighboringParticles
        Initial and neighboring particle configuration.
    timestep : float
        Time step used for numerical integration.
    flow_map_period : float
        Integration duration for each FTLE computation.
    num_ftles : int
        Number of FTLE computations to perform.
    num_snapshots : int
        Number of snapshots per flow map integration (derived from
        `flow_map_period / |timestep| + 1`).
    writer : Optional[BaseWriter]
        Writer object for saving output (if enabled).
    executor : ParallelExecutor
        Manager for executing FTLE computations in parallel.
    integrator : BaseIntegrator
        Time integrator created via :func:`create_integrator`.
    """

    def __init__(
        self,
        velocity_fn: Callable,  # TODO: improve this
        particles: NeighboringParticles,
        timestep: float,
        flow_map_period: float,
        num_ftles: int,
        integrator_name: str,  # TODO: improve this
        num_processes: int = 1,
        save_output: bool = False,
        output_format: str = "vtk",
        output_dir_name: Optional[str] = None,
    ):
        self.velocity_fn = velocity_fn
        self.particles = particles
        self.timestep = timestep
        self.flow_map_period = flow_map_period
        self.num_ftles = num_ftles
        self.num_snapshots = int(flow_map_period / abs(timestep)) + 1
        self.writer = None

        self.executor = ParallelExecutor(num_processes)

        interpolator = create_interpolator("analytical", velocity_fn=velocity_fn)

        self.integrator = create_integrator(integrator_name, interpolator)

        if self.timestep < 0:
            print("Running backward-time FTLE")
        else:
            print("Running forward-time FTLE")

        if save_output:
            if output_dir_name is None:
                now = datetime.now()
                output_dir_name = now.strftime("run-%Y-%m-%d-%Hh-%Mm-%Ss")
            self.writer = create_writer(output_format, output_dir_name)

    def _create_batches(self) -> list[BatchSource]:
        """
        Generate a list of analytical batch sources for each FTLE computation.

        Each batch corresponds to one FTLE field, containing the time sequence
        of integration snapshots based on the analytical velocity function.

        Returns
        -------
        list[BatchSource]
            List of :class:`AnalyticalBatchSource` objects, one per FTLE run.
        """
        # Start time for each FTLE batch
        start_times = np.arange(self.num_ftles) * self.timestep

        # Offsets within each batch
        offsets = np.arange(self.num_snapshots) * self.timestep

        # Broadcast addition to build all time batches
        time_batches = start_times[:, None] + offsets

        tasks: list[BatchSource] = []
        for i in range(self.num_ftles):
            task = AnalyticalBatchSource(
                self.velocity_fn,
                self.particles,
                self.timestep,
                time_batches[i],
            )
            tasks.append(task)
        return tasks

    def _worker(self, batch_source: BatchSource, progress_queue):
        """
        Worker routine executed in parallel processes to compute one FTLE field.

        This function instantiates an :class:`FTLESolver` for the given batch and
        runs it using the configured integrator and optional writer.

        Parameters
        ----------
        batch_source : BatchSource
            Batch source representing the time series for one FTLE computation.
        progress_queue : multiprocessing.Queue
            Queue for tracking progress among parallel workers.

        Returns
        -------
        np.ndarray or None
            The computed FTLE field as a NumPy array if `save_output` is False.
            Otherwise, returns None after writing the data to disk.
        """
        solver = FTLESolver(
            batch_source,
            integrator=self.integrator,
            progress_queue=progress_queue,
            output_writer=self.writer,
        )
        return solver.run()

    def run(self):
        """
        Execute all FTLE computations, either sequentially or in parallel.

        This method orchestrates the full FTLE workflow:
        batch generation → particle integration → flow map computation →
        FTLE evaluation → optional output writing.

        Returns
        -------
        np.ndarray or None
            - If `save_output` is False, returns an array of shape
              ``(num_ftles, n_points)`` containing the computed FTLE fields.
            - If `save_output` is True, returns None (fields are written to disk).

        Raises
        ------
        RuntimeError
            If no FTLE fields are returned due to worker failure.
        """

        batches = self._create_batches()
        results = self.executor.run(batches, self._worker)

        # Case 1: writer was used — results are all None
        if self.writer is not None:
            # Nothing to return; data already written to disk
            return

        # Case 2: no writer — results are np.ndarray (some may be None if a
        # worker failed)
        if not results:
            raise RuntimeError("No FTLE fields were returned (all results were None).")

        if len(results) == 1:
            return results[0]

        results = cast(list[np.ndarray], results)

        return np.stack(results, axis=0)  # (num_ftles, n_points)
