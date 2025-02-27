import os
from multiprocessing.managers import DictProxy
from queue import Queue
from typing import List

import numpy as np
from scipy.io import savemat
from tqdm import tqdm

from src.cauchy_green import compute_flow_map_jacobian
from src.file_readers import (
    read_seed_particles_coordinates,
)
from src.ftle import compute_ftle
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
        grid_files: List[str],
        particle_file: str,
        tqdm_position_queue: Queue[int],
        progress_dict: DictProxy,  # type: ignore
        interpolator_factory: InterpolatorFactory,
    ):
        self.index = index
        self.snapshot_files = snapshot_files
        self.grid_files = grid_files
        self.particle_file = particle_file
        self.progress_dict: DictProxy[int, bool] = progress_dict
        self.interpolator_factory = interpolator_factory
        self.tqdm_position_queue = tqdm_position_queue
        self.tqdm_position = None  # Will be assigned dynamically
        self.output_dir = f"outputs/{args.experiment_name}"

    def run(self) -> None:
        """Processes a single snapshot period."""
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

        for snapshot_file, grid_file in zip(self.snapshot_files, self.grid_files):
            tqdm_bar.set_description(f"FTLE {self.index:04d}: {snapshot_file}")
            tqdm_bar.update(1)

            interpolator = self.interpolator_factory.create_interpolator(
                snapshot_file, grid_file, args.interpolator
            )
            integrator.integrate(args.snapshot_timestep, particles, interpolator)

        self._compute_and_save_ftle(particles)

        tqdm_bar.clear()
        tqdm_bar.close()
        self.progress_dict[self.index] = True  # Notify progress monitor
        self.tqdm_position_queue.put(self.tqdm_position)

    def _compute_and_save_ftle(self, particles: NeighboringParticles) -> None:
        """Computes FTLE and saves the results."""
        jacobian = compute_flow_map_jacobian(particles)
        map_period = (len(self.snapshot_files) - 1) * abs(args.snapshot_timestep)
        ftle_field = compute_ftle(jacobian, map_period)
        ftle_field = np.array(ftle_field)  # enforce type compatibility in savemat

        os.makedirs(self.output_dir, exist_ok=True)

        filename = os.path.join(self.output_dir, f"ftle{self.index:04d}.mat")
        savemat(filename, {"ftle": ftle_field})
