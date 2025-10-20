from pathlib import Path
from typing import Callable, List, Optional, Protocol, Tuple

from src.file_readers import (
    read_coordinate,
    read_seed_particles_coordinates,
    read_velocity,
)
from src.interpolate import Interpolator
from src.my_types import ArrayFloat64Nx2, ArrayFloat64Nx3
from src.particles import NeighboringParticles


class BatchSource(Protocol):
    @property
    def timestep(self) -> float: ...
    @property
    def num_steps(self) -> int: ...
    @property
    def id(self) -> str: ...
    def get_particles(self) -> NeighboringParticles: ...
    def update_interpolator(
        self, interpolator: Interpolator, step_index: int
    ) -> None: ...


class FileBatchSource(BatchSource):
    def __init__(
        self,
        snapshot_files: List[str],
        coordinate_files: List[str],
        particle_file: str,
        snapshot_timestep: float,
        flow_map_period: int | float,
        grid_shape: Optional[tuple[int, ...]] = None,
    ):
        self.snapshot_files = snapshot_files
        self.coordinate_files = coordinate_files
        self.particle_file = particle_file  # Assume single file
        self.snapshot_timestep = snapshot_timestep
        self.flow_map_period = flow_map_period
        self.grid_shape = grid_shape
        self._n = len(snapshot_files)
        self._id = f"{Path(self.snapshot_files[0]).stem}"

    @property
    def id(self) -> str:
        return self._id

    @property
    def num_steps(self) -> int:
        return self._n

    @property
    def timestep(self) -> float:
        return self.snapshot_timestep

    def get_particles(self):
        return read_seed_particles_coordinates(self.particle_file)

    def get_data_for_step(
        self, step_index: int
    ) -> Tuple[ArrayFloat64Nx2 | ArrayFloat64Nx3, ArrayFloat64Nx2 | ArrayFloat64Nx3]:
        vel_file = self.snapshot_files[step_index]
        coord_file = self.coordinate_files[step_index]

        velocities = read_velocity(vel_file)
        coordinates = read_coordinate(coord_file)

        return velocities, coordinates

    def update_interpolator(self, interpolator: Interpolator, step_index: int) -> None:
        velocities, coordinates = self.get_data_for_step(step_index)
        interpolator.update(velocities, coordinates)


class AnalyticalBatchSource(BatchSource):
    def __init__(
        self,
        velocity_fn: Callable,  # TODO: add type
        particles: NeighboringParticles,
        timestep: float,
        times,  # TODO: add type -- 1D array of floats
    ):
        self.velocity_fn = velocity_fn
        self.particles = particles
        self._timestep = timestep
        self.times = times

    @property
    def id(self) -> str:
        return f"{self.times[0]:06f}"

    @property
    def num_steps(self) -> int:
        return len(self.times)

    @property
    def timestep(self) -> float:
        return self._timestep

    def get_particles(self):
        return self.particles

    def update_interpolator(self, interpolator, step_index: int) -> None:
        """AnalyticalInterpolator doesn’t need to be updated."""
        pass
