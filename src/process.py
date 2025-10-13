from pathlib import Path
from queue import Queue
from typing import Optional

import numpy as np

from dtos import FTLETask
from file_writers import FTLEWriter
from integrate import IntegratorStrategy
from my_types import ArrayFloat64N
from src.cauchy_green import (
    compute_flow_map_jacobian_2x2,
    compute_flow_map_jacobian_3x3,
)
from src.file_readers import (
    read_seed_particles_coordinates,
)
from src.ftle import compute_ftle_2x2, compute_ftle_3x3
from src.hyperparameters import args
from src.integrate import get_integrator
from src.interpolate import InterpolatorFactory
from src.particles import NeighboringParticles


class FTLESolver:
    """Computes the FTLE field given a batch of snapshots."""

    def __init__(
        self,
        batch_data: FTLETask,
        timestep: float,
        interpolator_factory: InterpolatorFactory,
        integrator: IntegratorStrategy,
        progress_queue: Queue,
        output_writer: Optional[FTLEWriter] = None,
    ):
        self.snapshot_files = batch_data["snapshots"]
        self.coordinate_files = batch_data["coordinates"]
        self.particle_file = batch_data["particles"]
        self.timestep = timestep
        self.interpolator_factory = interpolator_factory
        self.integrator = integrator
        self.output_writer = output_writer
        self.progress_queue = progress_queue

    def run(self) -> None:
        """Processes a single snapshot period."""

        # Assume a single particle file
        particles = read_seed_particles_coordinates(self.particle_file)

        for i, (snapshot_file, coordinate_file) in enumerate(
            zip(self.snapshot_files, self.coordinate_files), start=1
        ):
            interpolator = self.interpolator_factory.create_interpolator(
                snapshot_file, coordinate_file, args.interpolator
            )
            self.integrator.integrate(self.timestep, particles, interpolator)

            # publish progress: i goes from 1 .. len(self.snapshot_files)
            self.progress_queue.put((self._batch_name, i))

        # signal task done
        self.progress_queue.put((self._batch_name, "done"))

        ftle_field = self._compute_ftle(particles)

        if self.output_writer is not None:
            filename = "ftle_" + Path(self.snapshot_files[0]).stem
            self.output_writer.write(filename, ftle_field, particles.initial_centroid)

    def run_in_memory(
        self,
        particles: NeighboringParticles,
        snapshot_timestep: float,
        flow_map_period: float,
        integrator_name: str = "rk4",
    ):
        """
        Computes FTLE directly in memory (for Jupyter examples).

        Parameters
        ----------
        particles (NeighboringParticles): Particles to be tracked.
        snapshot_timestep (float): Time between consecutive snapshots.
        flow_map_period (float): Integration duration.
        integrator_name (str): e.g. 'rk4'.
        interpolator_name (str) e.g. 'cubic'.

        Returns
        -------
        ftle_field (np.ndarray): The computed FTLE field.
        particles (NeighboringParticles): Updated particle object (for visualization).
        """

        integrator = get_integrator(integrator_name)

        num_snapshots = int(flow_map_period / abs(snapshot_timestep)) + 1
        time_values = np.linspace(0.0, flow_map_period, num_snapshots) * np.sign(
            snapshot_timestep
        )

        for t in time_values:
            interpolator = self.interpolator_factory.create_interpolator_in_memory(
                time=t
            )
            integrator.integrate(snapshot_timestep, particles, interpolator)

        ftle_field = self._compute_ftle(particles)

        return ftle_field, particles

    def _compute_ftle(self, particles: NeighboringParticles) -> ArrayFloat64N:
        """Computes FTLE and saves the results."""

        if particles.num_neighbors == 4:
            jacobian = compute_flow_map_jacobian_2x2(particles)
            map_period = (len(self.snapshot_files) - 1) * abs(args.snapshot_timestep)
            return compute_ftle_2x2(jacobian, map_period)

        jacobian = compute_flow_map_jacobian_3x3(particles)
        map_period = (len(self.snapshot_files) - 1) * abs(args.snapshot_timestep)
        return compute_ftle_3x3(jacobian, map_period)

    @property
    def _batch_name(self):
        return f"{Path(self.snapshot_files[0]).stem}"
