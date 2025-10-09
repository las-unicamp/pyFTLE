from multiprocessing.managers import DictProxy
from queue import Queue
from typing import List

from tqdm import tqdm

from file_writers import FTLEWriter
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
        tqdm_position_queue: Queue[int],
        progress_dict: DictProxy,  # type: ignore
        interpolator_factory: InterpolatorFactory,
        output_writer: FTLEWriter,
    ):
        self.index = index
        self.snapshot_files = snapshot_files
        self.coordinate_files = coordinate_files
        self.particle_file = particle_file
        self.progress_dict: DictProxy[int, bool] = progress_dict
        self.interpolator_factory = interpolator_factory
        self.tqdm_position_queue = tqdm_position_queue
        self.tqdm_position = None  # Will be assigned dynamically
        self.output_writer = output_writer

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

        for snapshot_file, coordinate_file in zip(
            self.snapshot_files, self.coordinate_files
        ):
            tqdm_bar.set_description(f"FTLE {self.index:04d}: {snapshot_file}")
            tqdm_bar.update(1)

            interpolator = self.interpolator_factory.create_interpolator(
                snapshot_file, coordinate_file, args.interpolator
            )
            integrator.integrate(args.snapshot_timestep, particles, interpolator)

        self._compute_and_save_ftle(particles)

        tqdm_bar.clear()
        tqdm_bar.close()
        self.progress_dict[self.index] = True  # Notify progress monitor
        self.tqdm_position_queue.put(self.tqdm_position)

    def _compute_and_save_ftle(self, particles: NeighboringParticles) -> None:
        """Computes FTLE and saves the results."""

        if particles.num_neighbors == 4:
            jacobian = compute_flow_map_jacobian_2x2(particles)
            map_period = (len(self.snapshot_files) - 1) * abs(args.snapshot_timestep)
            ftle_field = compute_ftle_2x2(jacobian, map_period)

        else:
            jacobian = compute_flow_map_jacobian_3x3(particles)
            map_period = (len(self.snapshot_files) - 1) * abs(args.snapshot_timestep)
            ftle_field = compute_ftle_3x3(jacobian, map_period)

        filename = f"ftle{self.index:04d}"
        self.output_writer.write(filename, ftle_field, particles.initial_centroid)
