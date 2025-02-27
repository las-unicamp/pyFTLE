import numpy as np
from numba import njit  # type: ignore

from src.my_types import ArrayFloat64N4x2, ArrayFloat64Nx2, ArrayFloat64Nx2x2
from src.particles import NeighboringParticles


@njit
def _compute_flow_map_jacobian_in_numba(
    particles_positions: ArrayFloat64N4x2,
    delta_right_left: ArrayFloat64Nx2,
    initial_delta_right_left: ArrayFloat64Nx2,
    delta_top_bottom: ArrayFloat64Nx2,
    initial_delta_top_bottom: ArrayFloat64Nx2,
) -> ArrayFloat64Nx2x2:
    """
    Compute the flow map Jacobian (deformation gradient) based on the the initial
    and deformed positions.

    Args:
    - particles_positions (ArrayFloat64Nx2): The positions at forward or backward
          time.
    - delta_right_left (ArrayFloat64Nx2): Change in right-left positions.
    - initial_delta_right_left (ArrayFloat64Nx2): Initial change in right-left
          positions.
    - delta_top_bottom (ArrayFloat64Nx2): Change in top-bottom positions.
    - initial_delta_top_bottom (ArrayFloat64Nx2): Initial change in top-bottom
          positions.

    Returns:
    - jacobian (ArrayFloat64Nx2x2): The flow map Jacobian.
    """
    num_particles = particles_positions.shape[0] // 4  # Number of particle groups (N)
    jacobian = np.empty((num_particles, 2, 2))

    for i in range(num_particles):
        jacobian[i, 0, 0] = delta_right_left[i, 0] / initial_delta_right_left[i, 0]
        jacobian[i, 0, 1] = delta_top_bottom[i, 0] / initial_delta_top_bottom[i, 1]
        jacobian[i, 1, 0] = delta_right_left[i, 1] / initial_delta_right_left[i, 0]
        jacobian[i, 1, 1] = delta_top_bottom[i, 1] / initial_delta_top_bottom[i, 1]

    return jacobian


# Wrapper function for the NeighboringParticles dataclass
def compute_flow_map_jacobian(
    particles: NeighboringParticles,
) -> ArrayFloat64Nx2x2:
    return _compute_flow_map_jacobian_in_numba(
        particles.positions,
        particles.delta_right_left,
        particles.initial_delta_right_left,
        particles.delta_top_bottom,
        particles.initial_delta_top_bottom,
    )
