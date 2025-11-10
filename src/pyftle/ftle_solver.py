from queue import Queue
from typing import Optional

from pyftle.cauchy_green import (
    compute_flow_map_jacobian_2x2,
    compute_flow_map_jacobian_3x3,
)
from pyftle.data_source import BatchSource
from pyftle.file_writers import FTLEWriter
from pyftle.ftle import compute_ftle_2x2, compute_ftle_3x3
from pyftle.integrate import Integrator
from pyftle.my_types import ArrayN


class FTLESolver:
    """
    Compute the Finite-Time Lyapunov Exponent (FTLE) field from a sequence of
    velocity data files.

    The solver integrates particle trajectories over a given flow map period,
    computes the flow map Jacobian, and evaluates the FTLE field either in 2D
    or 3D. Data are processed sequentially from a :class:`BatchSource`, using
    an :class:`Integrator` for time integration, and optionally writing results
    through a :class:`FTLEWriter`.

    Parameters
    ----------
    source : BatchSource
        Source of velocity field data and particle initialization. Must provide
        velocity data, interpolation updates, and particle positions for each
        time step.
    integrator : Integrator
        Object responsible for advancing particle positions in time using the
        current velocity interpolator.
    progress_queue : Optional[Queue], default=None
        Optional multiprocessing queue for publishing progress updates during
        FTLE computation. Each iteration sends a tuple ``(id, i)`` and a final
        ``(id, "done")`` message when completed.
    output_writer : Optional[FTLEWriter], default=None
        Writer used to save the computed FTLE field to disk. If not provided,
        the computed field is returned instead of written.

    Notes
    -----
    The `FTLESolver` assumes that the sequence of velocity field files in
    `source` is ordered correctly for either forward or backward FTLE mapping.
    It is the user's responsibility to ensure temporal consistency in the input
    data.

    Examples
    --------
    >>> solver = FTLESolver(source, integrator, output_writer=writer)
    >>> solver.run()  # Computes and writes FTLE field
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
        """
        Run the FTLE computation for the current flow map period.

        This method performs the full FTLE workflow:
        1. Loads particle positions from the data source.
        2. Sequentially integrates all time steps in the dataset.
        3. Publishes progress updates to the queue (if provided).
        4. Computes the FTLE field using the final particle configuration.
        5. Writes or returns the FTLE field.

        Returns
        -------
        Optional[ArrayN]
            The computed FTLE field if `output_writer` is not provided;
            otherwise, returns ``None`` after writing the results to disk.

        Notes
        -----
        The time integration is performed in-place using the provided
        :class:`Integrator`, which updates particle positions over
        ``num_steps`` iterations. After all steps, the flow map Jacobian
        is evaluated to compute the FTLE.
        """

        self.particles = self.source.get_particles()

        id = self.source.id
        timestep = self.source.timestep
        num_steps = self.source.num_steps

        for i in range(num_steps):
            self.source.update_interpolator(self.integrator.interpolator, i)

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
            return ftle_field

    def _compute_ftle(self) -> ArrayN:
        r"""
        Compute the FTLE field from the final particle configuration.

        Depending on the number of neighboring particles, the method
        automatically selects the appropriate dimensional version (2x2 or 3x3)
        of the flow map Jacobian and FTLE computation.

        Returns
        -------
        ftle_field : ArrayN
            The computed FTLE scalar field, one value per particle centroid.

        Notes
        -----
        The flow map Jacobian is computed using:
            * :func:`compute_flow_map_jacobian_2x2` for 2D datasets.
            * :func:`compute_flow_map_jacobian_3x3` for 3D datasets.

        The FTLE is then obtained as:

            .. math::

                \\text{FTLE} = \\frac{1}{|T|} \\ln \\sqrt{\\lambda_{max}(C)}

        where :math:`T` is the flow map period and
        :math:`\\lambda_{max}(C)` is the largest eigenvalue of the
        Cauchy-Green deformation tensor :math:`C = F^\\top F`.
        """
        num_steps = self.source.num_steps
        timestep = self.source.timestep

        if self.particles.num_neighbors == 4:
            jacobian = compute_flow_map_jacobian_2x2(self.particles)
            map_period = (num_steps - 1) * abs(timestep)
            return compute_ftle_2x2(jacobian, map_period)

        jacobian = compute_flow_map_jacobian_3x3(self.particles)
        map_period = (num_steps - 1) * abs(timestep)
        return compute_ftle_3x3(jacobian, map_period)
