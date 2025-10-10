from multiprocessing.managers import DictProxy
from queue import Queue
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from file_writers import FTLEWriter
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


class SnapshotProcessor:
    """Handles the computation of FTLE for a single snapshot period."""

    def __init__(
        self,
        index: int,
        snapshot_files: List[str],
        coordinate_files: List[str],
        particle_file: str,
        interpolator_factory: InterpolatorFactory,
        output_writer: Optional[FTLEWriter] = None,
        tqdm_position_queue: Optional[Queue[int]] = None,
        progress_dict: Optional[DictProxy[int, bool]] = None,  # type: ignore
    ):
        self.index = index
        self.snapshot_files = snapshot_files
        self.coordinate_files = coordinate_files
        self.particle_file = particle_file
        self.progress_dict = progress_dict
        self.interpolator_factory = interpolator_factory
        self.tqdm_position_queue = tqdm_position_queue
        self.tqdm_position = None  # Will be assigned dynamically
        self.output_writer = output_writer

    def run(self) -> None:
        """Processes a single snapshot period."""
        if self.tqdm_position_queue is None or self.progress_dict is None:
            raise RuntimeError(
                "SnapshotProcessor.run() requires multiprocessing context."
            )

        self.tqdm_position = self.tqdm_position_queue.get()

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

        for snapshot_file, coordinate_file in zip(
            self.snapshot_files, self.coordinate_files
        ):
            tqdm_bar.set_description(f"FTLE {self.index:04d}: {snapshot_file}")
            tqdm_bar.update(1)

            interpolator = self.interpolator_factory.create_interpolator(
                snapshot_file, coordinate_file, args.interpolator
            )
            integrator.integrate(args.snapshot_timestep, particles, interpolator)

        ftle_field = self._compute_ftle(particles)

        if self.output_writer is not None:
            filename = f"ftle{self.index:04d}"
            self.output_writer.write(filename, ftle_field, particles.initial_centroid)

        tqdm_bar.clear()
        tqdm_bar.close()
        self.progress_dict[self.index] = True  # Notify progress monitor
        self.tqdm_position_queue.put(self.tqdm_position)

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
