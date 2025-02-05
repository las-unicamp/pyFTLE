import numpy as np

from src.my_types import ArrayFloat32Nx2x2
from src.particle_position import PositionDict


def compute_flow_map_jacobian(particles: PositionDict) -> ArrayFloat32Nx2x2:
    """
    Compute the flow map Jacobian (deformation gradient) based on the the initial
    and deformed positions.

    Args:
    - particles (PositionDict): The positions at the forward or backward time.

    Returns:
    - np.ndarray: The Cauchy-Green deformation tensor.
    """
    num_particles = len(particles)
    jacobian = np.empty((num_particles, 2, 2))

    jacobian[:, 0, 0] = (
        particles.data.delta_right_left[:, 0]
        / particles.data.initial_delta_right_left[:, 0]
    )

    jacobian[:, 0, 1] = (
        particles.data.delta_top_bottom[:, 0]
        / particles.data.initial_delta_top_bottom[:, 1]
    )
    jacobian[:, 1, 0] = (
        particles.data.delta_right_left[:, 1]
        / particles.data.initial_delta_right_left[:, 0]
    )

    jacobian[:, 1, 1] = (
        particles.data.delta_top_bottom[:, 1]
        / particles.data.initial_delta_top_bottom[:, 1]
    )

    return jacobian
