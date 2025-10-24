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
    A notebook-friendly FTLE manager for in-memory data.

    This manager batches and runs the FTLE solver for in-memory data.
    """

    # TODO: improve docstring

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
        """Generate overlapping batches of snapshot, coordinate and particle files"""
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
        """Wrapper function for parallel execution."""
        solver = FTLESolver(
            batch_source,
            integrator=self.integrator,
            progress_queue=progress_queue,
            output_writer=self.writer,
        )
        return solver.run()

    def run(self):
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
