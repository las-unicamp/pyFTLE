"""
Batch data sources for FTLE computation.

This module defines the `BatchSource` protocol and concrete implementations
(`FileBatchSource` and `AnalyticalBatchSource`) for providing particle and
velocity field data during flow map integration.

Two main use cases are supported:
1. File-based data: reading velocity and coordinate snapshots from disk.
2. Analytical data: using a user-defined velocity function evaluated at runtime.

All sources expose a consistent interface defined by the `BatchSource` protocol,
so they can be used interchangeably in higher-level FTLE computation routines.
"""

from pathlib import Path
from typing import Callable, List, Protocol, Tuple

from pyftle.file_readers import (
    read_coordinate,
    read_seed_particles_coordinates,
    read_velocity,
)
from pyftle.interpolate import Interpolator
from pyftle.my_types import Array2xN, Array3xN
from pyftle.particles import NeighboringParticles


class BatchSource(Protocol):
    """
    Protocol defining the interface for batch data sources used in FTLE computation.

    Any concrete implementation must provide methods and properties to:
    - Access the time-step information.
    - Retrieve initial particle positions.
    - Update the velocity interpolator for a given snapshot.

    This abstraction allows using either numerical data (from files) or analytical
    functions for flow map integration.
    """

    @property
    def timestep(self) -> float:
        """Time step between consecutive snapshots or time samples."""
        ...

    @property
    def num_steps(self) -> int:
        """Total number of available time steps."""
        ...

    @property
    def id(self) -> str:
        """
        Unique identifier for the batch source (e.g., first file name or time
        label).
        """
        ...

    def get_particles(self) -> NeighboringParticles:
        """
        Retrieve the set of neighboring particles used for flow map computation.

        Returns
        -------
        NeighboringParticles
            Dataclass containing particle positions and their initial displacements.
        """
        ...

    def update_interpolator(self, interpolator: Interpolator, step_index: int) -> None:
        """
        Update the velocity interpolator with the data for the given time step.

        Parameters
        ----------
        interpolator : Interpolator
            The interpolator instance to update with velocity (and optionally
            coordinate) data.
        step_index : int
            Index of the current time step (0-based).
        """
        ...


class FileBatchSource(BatchSource):
    """
    Batch source that loads velocity and coordinate data from files.

    This implementation supports both static and time-varying coordinate grids.
    It reads the velocity and coordinate snapshots corresponding to each time step,
    as well as the seed particle positions for flow map initialization.

    Parameters
    ----------
    snapshot_files : list of str
        List of file paths containing velocity field data for each time step.
    coordinate_files : list of str
        List of file paths containing the spatial grid coordinates. Can be identical
        for all time steps if the grid is stationary.
    particle_file : str
        Path to the file containing seed particle positions.
    snapshot_timestep : float
        Time step between consecutive velocity snapshots.
    flow_map_period : int or float
        Number of time steps or total time duration between flow map evaluations.

    Attributes
    ----------
    snapshot_files : list of str
        Paths to velocity data files.
    coordinate_files : list of str
        Paths to coordinate grid files.
    particle_file : str
        Path to the seed particle file.
    snapshot_timestep : float
        Time step between snapshots.
    flow_map_period : int or float
        Integration period between flow map outputs.
    reuse_coordinates : bool
        Whether the same coordinate file is used for all time steps.
    """

    def __init__(
        self,
        snapshot_files: List[str],
        coordinate_files: List[str],
        particle_file: str,
        snapshot_timestep: float,
        flow_map_period: int | float,
    ):
        self.snapshot_files = snapshot_files
        self.coordinate_files = coordinate_files
        self.particle_file = particle_file  # Assume single file
        self.snapshot_timestep = snapshot_timestep
        self.flow_map_period = flow_map_period
        self._n = len(snapshot_files)
        self._id = f"{Path(self.snapshot_files[0]).stem}"

        is_coordinate_files_identical = len(set(coordinate_files)) == 1
        self.reuse_coordinates = is_coordinate_files_identical

    @property
    def id(self) -> str:
        """Unique identifier derived from the first snapshot file name."""
        return self._id

    @property
    def num_steps(self) -> int:
        """Number of available time steps (number of snapshot files)."""
        return self._n

    @property
    def timestep(self) -> float:
        """Time interval between consecutive velocity snapshots."""
        return self.snapshot_timestep

    def get_particles(self) -> NeighboringParticles:
        """
        Load and return the seed particles used to initialize the flow map.

        Returns
        -------
        NeighboringParticles
            Dataclass containing particle positions and neighbor offsets.
        """
        return read_seed_particles_coordinates(self.particle_file)

    def get_data_for_step(
        self, step_index: int
    ) -> Tuple[Array2xN | Array3xN, Array2xN | Array3xN | None]:
        """
        Load velocity and coordinate data for a specific time step.

        Parameters
        ----------
        step_index : int
            Index of the time step to read (0-based).

        Returns
        -------
        velocities : Array2xN or Array3xN
            Velocity field data at the current time step.
        coordinates : Array2xN or Array3xN or None
            Coordinate grid corresponding to the current time step.
            Returns `None` if the grid is reused and unchanged.
        """
        vel_file = self.snapshot_files[step_index]
        coord_file = self.coordinate_files[step_index]

        velocities = read_velocity(vel_file)
        coordinates = None

        if step_index == 0 or not self.reuse_coordinates:
            coordinates = read_coordinate(coord_file)

        return velocities, coordinates

    def update_interpolator(self, interpolator: Interpolator, step_index: int) -> None:
        """
        Update the velocity interpolator with the data for the specified time step.

        Parameters
        ----------
        interpolator : Interpolator
            Interpolator instance to update.
        step_index : int
            Index of the time step to load.
        """
        velocities, coordinates = self.get_data_for_step(step_index)
        interpolator.update(velocities, coordinates)


class AnalyticalBatchSource(BatchSource):
    """
    Batch source for analytical velocity fields.

    This implementation allows using a user-defined velocity function instead of
    reading data from files. It provides access to precomputed particle positions
    and a sequence of time values.

    Parameters
    ----------
    velocity_fn : Callable
        Analytical function that returns the velocity field given space and time.
        Expected signature:
        `velocity_fn(x: Array2xN | Array3xN, t: float) -> Array2xN | Array3xN`
    particles : NeighboringParticles
        Particle data used for initialization.
    timestep : float
        Time step between consecutive evaluations.
    times : array-like of float
        Sequence of time values at which the velocity function is evaluated.
    """

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
        """Unique identifier based on the initial time value."""
        return f"{self.times[0]:06f}"

    @property
    def num_steps(self) -> int:
        """Number of available time samples."""
        return len(self.times)

    @property
    def timestep(self) -> float:
        """Fixed time step between consecutive evaluations."""
        return self._timestep

    def get_particles(self) -> NeighboringParticles:
        """
        Return the particle data associated with this analytical source.

        Returns
        -------
        NeighboringParticles
            Dataclass containing particle positions and neighbor offsets.
        """
        return self.particles

    def update_interpolator(self, interpolator, step_index: int) -> None:
        """
        No-op method: analytical velocity fields do not require data updates.

        Parameters
        ----------
        interpolator : Interpolator
            The interpolator instance (unused).
        step_index : int
            Index of the current step (unused).
        """
        pass
