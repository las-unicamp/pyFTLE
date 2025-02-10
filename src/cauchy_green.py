import numpy as np

from src.my_types import ArrayFloat32Nx2x2
from src.particles import NeighboringParticles


def compute_flow_map_jacobian(particles: NeighboringParticles) -> ArrayFloat32Nx2x2:
    """
    Compute the flow map Jacobian (deformation gradient) based on the the initial
    and deformed positions.

    Args:
    - particles (NeighboringParticles): The positions at forward or backward time.

    Returns:
    - jacobian (ArrayFloat32Nx2x2): The flow map Jacobian.
    """
    num_particles = len(particles)
    jacobian = np.empty((num_particles, 2, 2))

    jacobian[:, 0, 0] = (
        particles.delta_right_left[:, 0] / particles.initial_delta_right_left[:, 0]
    )

    jacobian[:, 0, 1] = (
        particles.delta_top_bottom[:, 0] / particles.initial_delta_top_bottom[:, 1]
    )
    jacobian[:, 1, 0] = (
        particles.delta_right_left[:, 1] / particles.initial_delta_right_left[:, 0]
    )

    jacobian[:, 1, 1] = (
        particles.delta_top_bottom[:, 1] / particles.initial_delta_top_bottom[:, 1]
    )

    return jacobian
