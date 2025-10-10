from typing import Callable, Optional

import numpy as np

from particles import NeighboringParticles
from src.file_writers import create_writer
from src.interpolate import InterpolatorFactory
from src.process import SnapshotProcessor


class FTLESolver:
    def __init__(
        self,
        velocity_field: Callable | np.ndarray,
        grid: np.ndarray,
        particles: NeighboringParticles,
        snapshot_timestep: float,
        flow_map_period: float,
        integrator: str = "rk4",
        interpolator: str = "cubic",
        num_processes: int = 1,
        output_format: Optional[str] = None,
    ):
        """
        A notebook-friendly FTLE solver.

        Parameters
        ----------
        velocity_field (callable or ndarray): Either a function
            f(x, y, z, t) -> (u, v, w)
            or an array of shape (num_snapshots, Nx, Ny, Nz, 3)
        grid (tuple of ndarrays): (X, Y, Z) coordinate arrays.
        particles (ndarray): Particle positions to be tracked.
        snapshot_timestep (float): Time between snapshots.
        flow_map_period (float): Duration over which the flow map is integrated.
        integrator, interpolator (str): Same options as the CLI version.
        num_processes (int): Number of processes to use (default: 1).
        output_format (str or None): If given (e.g. 'vtk'), saves output.
            Otherwise, returns the FTLE field.
        """
        self.velocity_field = velocity_field
        self.grid = grid
        self.particles = particles
        self.snapshot_timestep = snapshot_timestep
        self.flow_map_period = flow_map_period
        self.integrator = integrator
        self.interpolator = interpolator
        self.num_processes = num_processes
        self.output_format = output_format

    def run(self):
        # 1. Construct the in-memory interpolator factory
        interpolator_factory = InterpolatorFactory.from_memory(
            self.velocity_field, self.grid
        )

        # 2. Optional writer
        writer = None
        if self.output_format is not None:
            writer = create_writer(
                self.output_format,
                "outputs/in_memory_example",
                grid_shape=self.grid.shape,
            )

        # 3. Create and run the processor
        processor = SnapshotProcessor(
            index=0,
            snapshot_files=[],  # not used in-memory
            coordinate_files=[],  # not used in-memory
            particle_file="",  # not used in-memory
            interpolator_factory=interpolator_factory,
            output_writer=writer,
        )

        # 3. Run single-snapshot FTLE computation (directly)
        ftle, particles = processor.run_in_memory(
            particles=self.particles,
            snapshot_timestep=self.snapshot_timestep,
            flow_map_period=self.flow_map_period,
            integrator_name=self.integrator,
        )

        return ftle, particles
