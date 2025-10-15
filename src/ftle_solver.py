from queue import Queue
from typing import Optional

from src.cauchy_green import (
    compute_flow_map_jacobian_2x2,
    compute_flow_map_jacobian_3x3,
)
from src.data_source import BatchSource
from src.file_writers import FTLEWriter
from src.ftle import compute_ftle_2x2, compute_ftle_3x3
from src.integrate import Integrator
from src.my_types import ArrayFloat64N


class FTLESolver:
    """
    Computes the FTLE field from a batch of data.

    All velocity and coordinate data provided in `source` will be processed
    sequentially. Therefore, it is important to pre-select these files carefully
    so that they cover the desired flow map period. Also, the ordering of
    these data must be taken care properly for the correct forward/backward
    FTLE mapping.
    """

    def __init__(
        self,
        source: BatchSource,
        integrator: Integrator,
        progress_queue: Optional[Queue] = None,
        output_writer: Optional[FTLEWriter] = None,
    ):
        self.source = source
        self.integrator = integrator
        self.output_writer = output_writer
        self.progress_queue = progress_queue

    def run(self):  # TODO: add return type
        """Processes a single snapshot period."""

        self.particles = self.source.get_particles()

        id = self.source.id
        timestep = self.source.timestep
        num_steps = self.source.num_steps

        for i in range(num_steps):
            velocities, coordinates = self.source.get_data_for_step(i)

            self.integrator.interpolator.update(velocities, coordinates)
            self.integrator.integrate(timestep, self.particles)

            # publish progress: i goes from 1 ... num_steps
            if self.progress_queue:
                self.progress_queue.put((id, i))

        if self.progress_queue:
            # signal task done
            self.progress_queue.put((id, "done"))

        ftle_field = self._compute_ftle()

        if self.output_writer is not None:
            filename = f"ftle_{id}"
            self.output_writer.write(
                filename, ftle_field, self.particles.initial_centroid
            )
        else:
            return ftle_field, self.particles

    def _compute_ftle(self) -> ArrayFloat64N:
        """Computes FTLE and saves the results."""
        num_steps = self.source.num_steps
        timestep = self.source.timestep

        if self.particles.num_neighbors == 4:
            jacobian = compute_flow_map_jacobian_2x2(self.particles)
            map_period = (num_steps - 1) * abs(timestep)
            return compute_ftle_2x2(jacobian, map_period)

        jacobian = compute_flow_map_jacobian_3x3(self.particles)
        map_period = (num_steps - 1) * abs(timestep)
        return compute_ftle_3x3(jacobian, map_period)
