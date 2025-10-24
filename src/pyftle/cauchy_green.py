import numpy as np
from numba import njit  # type: ignore

from pyftle.my_types import (
    ArrayFloat64N4x2,
    ArrayFloat64N6x3,
    ArrayFloat64Nx2,
    ArrayFloat64Nx2x2,
    ArrayFloat64Nx3,
    ArrayFloat64Nx3x3,
)
from pyftle.particles import NeighboringParticles


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


@njit
def _compute_flow_map_jacobian_in_numba_3x3(
    particles_positions: ArrayFloat64N6x3,
    delta_right_left: ArrayFloat64Nx3,
    initial_delta_right_left: ArrayFloat64Nx3,
    delta_top_bottom: ArrayFloat64Nx3,
    initial_delta_top_bottom: ArrayFloat64Nx3,
    delta_front_back: ArrayFloat64Nx3,
    initial_delta_front_back: ArrayFloat64Nx3,
) -> ArrayFloat64Nx3x3:
    """
    Compute the flow map Jacobian (deformation gradient) based on the the initial
    and deformed positions.

    Args:
    - particles_positions (ArrayFloat64Nx3): The positions at forward or backward
          time.
    - delta_right_left (ArrayFloat64Nx3): Change in right-left positions.
    - initial_delta_right_left (ArrayFloat64Nx3): Initial change in right-left
          positions.
    - delta_top_bottom (ArrayFloat64Nx3): Change in top-bottom positions.
    - initial_delta_top_bottom (ArrayFloat64Nx3): Initial change in top-bottom
          positions.
    - delta_front_back (ArrayFloat64Nx3): Change in front-back positions.
    - initial_delta_front_back (ArrayFloat64Nx3): Initial change in front-back
          positions.


    Returns:
    - jacobian (ArrayFloat64Nx3x3): The flow map Jacobian.
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
) -> ArrayFloat64Nx2x2:
    # print('here')
    return _compute_flow_map_jacobian_in_numba(
        particles.positions,
        particles.delta_right_left,
        particles.initial_delta_right_left,
        particles.delta_top_bottom,
        particles.initial_delta_top_bottom,
    )


def compute_flow_map_jacobian_3x3(
    particles: NeighboringParticles,
) -> ArrayFloat64Nx3x3:
    return _compute_flow_map_jacobian_in_numba_3x3(
        particles.positions,
        particles.delta_right_left,
        particles.initial_delta_right_left,
        particles.delta_top_bottom,
        particles.initial_delta_top_bottom,
        particles.delta_front_back,
        particles.initial_delta_front_back,
    )
