import numpy as np
from numba import njit  # type: ignore

from pyftle.my_types import (
    Array4Nx2,
    Array6Nx3,
    ArrayNx2,
    ArrayNx2x2,
    ArrayNx3,
    ArrayNx3x3,
)
from pyftle.particles import NeighboringParticles


@njit
def _compute_flow_map_jacobian_in_numba(
    particles_positions: Array4Nx2,
    delta_right_left: ArrayNx2,
    initial_delta_right_left: ArrayNx2,
    delta_top_bottom: ArrayNx2,
    initial_delta_top_bottom: ArrayNx2,
) -> ArrayNx2x2:
    """
    Compute the flow map Jacobian (deformation gradient) based on the the initial
    and deformed positions.

    Args:
    - particles_positions (ArrayNx2): The positions at forward or backward
          time.
    - delta_right_left (ArrayNx2): Change in right-left positions.
    - initial_delta_right_left (ArrayNx2): Initial change in right-left
          positions.
    - delta_top_bottom (ArrayNx2): Change in top-bottom positions.
    - initial_delta_top_bottom (ArrayNx2): Initial change in top-bottom
          positions.

    Returns:
    - jacobian (ArrayNx2x2): The flow map Jacobian.
    """
    num_particles = particles_positions.shape[0] // 4  # Number of particle groups (N)
    jacobian = np.empty((num_particles, 2, 2))

    for i in range(num_particles):
        jacobian[i, 0, 0] = delta_right_left[i, 0] / initial_delta_right_left[i, 0]
        jacobian[i, 0, 1] = delta_top_bottom[i, 0] / initial_delta_top_bottom[i, 1]
        jacobian[i, 1, 0] = delta_right_left[i, 1] / initial_delta_right_left[i, 0]
        jacobian[i, 1, 1] = delta_top_bottom[i, 1] / initial_delta_top_bottom[i, 1]

    return jacobian


@njit
def _compute_flow_map_jacobian_in_numba_3x3(
    particles_positions: Array6Nx3,
    delta_right_left: ArrayNx3,
    initial_delta_right_left: ArrayNx3,
    delta_top_bottom: ArrayNx3,
    initial_delta_top_bottom: ArrayNx3,
    delta_front_back: ArrayNx3,
    initial_delta_front_back: ArrayNx3,
) -> ArrayNx3x3:
    """
    Compute the flow map Jacobian (deformation gradient) based on the the initial
    and deformed positions.

    Args:
    - particles_positions (ArrayNx3): The positions at forward or backward
          time.
    - delta_right_left (ArrayNx3): Change in right-left positions.
    - initial_delta_right_left (ArrayNx3): Initial change in right-left
          positions.
    - delta_top_bottom (ArrayNx3): Change in top-bottom positions.
    - initial_delta_top_bottom (ArrayNx3): Initial change in top-bottom
          positions.
    - delta_front_back (ArrayNx3): Change in front-back positions.
    - initial_delta_front_back (ArrayNx3): Initial change in front-back
          positions.


    Returns:
    - jacobian (ArrayNx3x3): The flow map Jacobian.
    """
    num_particles = particles_positions.shape[0] // 6  # Number of particle groups (N)
    jacobian = np.empty((num_particles, 3, 3))

    for i in range(num_particles):
        jacobian[i, 0, 0] = delta_right_left[i, 0] / initial_delta_right_left[i, 0]
        jacobian[i, 0, 1] = delta_top_bottom[i, 0] / initial_delta_top_bottom[i, 1]
        jacobian[i, 0, 2] = delta_front_back[i, 0] / initial_delta_front_back[i, 2]

        jacobian[i, 1, 0] = delta_right_left[i, 1] / initial_delta_right_left[i, 0]
        jacobian[i, 1, 1] = delta_top_bottom[i, 1] / initial_delta_top_bottom[i, 1]
        jacobian[i, 1, 2] = delta_front_back[i, 1] / initial_delta_front_back[i, 2]

        jacobian[i, 2, 0] = delta_right_left[i, 2] / initial_delta_right_left[i, 0]
        jacobian[i, 2, 1] = delta_top_bottom[i, 2] / initial_delta_top_bottom[i, 1]
        jacobian[i, 2, 2] = delta_front_back[i, 2] / initial_delta_front_back[i, 2]

    return jacobian


# Wrapper function for the NeighboringParticles dataclass
def compute_flow_map_jacobian_2x2(
    particles: NeighboringParticles,
) -> ArrayNx2x2:
    return _compute_flow_map_jacobian_in_numba(
        particles.positions,
        particles.delta_right_left,
        particles.initial_delta_right_left,
        particles.delta_top_bottom,
        particles.initial_delta_top_bottom,
    )


def compute_flow_map_jacobian_3x3(
    particles: NeighboringParticles,
) -> ArrayNx3x3:
    return _compute_flow_map_jacobian_in_numba_3x3(
        particles.positions,
        particles.delta_right_left,
        particles.initial_delta_right_left,
        particles.delta_top_bottom,
        particles.initial_delta_top_bottom,
        particles.delta_front_back,
        particles.initial_delta_front_back,
    )
