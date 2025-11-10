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
    Compute the 2D flow map Jacobian (deformation gradient) using the positions
    of four neighboring particles.

    This function evaluates the spatial derivatives of the flow map based on
    the displacement between initially neighboring particles. It is meant to be
    used internally with Numba for performance.

    Parameters
    ----------
    particles_positions : Array4Nx2
        Positions of all particles at the current (forward or backward) time.
        Each group of four consecutive particles corresponds to one seed point
        and its right, left, top, and bottom neighbors.
    delta_right_left : ArrayNx2
        Vector difference between right and left neighboring particle positions
        at the current time.
    initial_delta_right_left : ArrayNx2
        Initial vector difference between right and left neighboring particle
        positions.
    delta_top_bottom : ArrayNx2
        Vector difference between top and bottom neighboring particle positions
        at the current time.
    initial_delta_top_bottom : ArrayNx2
        Initial vector difference between top and bottom neighboring particle
        positions.

    Returns
    -------
    jacobian : ArrayNx2x2
        The flow map Jacobian (deformation gradient) tensor for each particle
        group. Each tensor represents the local spatial derivative of the flow
        map with respect to the initial configuration.

    Notes
    -----
    The Jacobian is computed as:

        J = [ Δx_right_left / ΔX_right_left,
              Δx_top_bottom  / ΔX_top_bottom ]

    where Δx and ΔX are the current and initial separation vectors between
    neighboring particles.
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
    Compute the 3D flow map Jacobian (deformation gradient) using the positions
    of six neighboring particles.

    This function evaluates the spatial derivatives of the 3D flow map based on
    displacements between initially neighboring particles. It is intended for
    internal use with Numba acceleration.

    Parameters
    ----------
    particles_positions : Array6Nx3
        Positions of all particles at the current (forward or backward) time.
        Each group of six consecutive particles corresponds to one seed point
        and its right, left, top, bottom, front, and back neighbors.
    delta_right_left : ArrayNx3
        Vector difference between right and left neighboring particle positions
        at the current time.
    initial_delta_right_left : ArrayNx3
        Initial vector difference between right and left neighboring particle
        positions.
    delta_top_bottom : ArrayNx3
        Vector difference between top and bottom neighboring particle positions
        at the current time.
    initial_delta_top_bottom : ArrayNx3
        Initial vector difference between top and bottom neighboring particle
        positions.
    delta_front_back : ArrayNx3
        Vector difference between front and back neighboring particle positions
        at the current time.
    initial_delta_front_back : ArrayNx3
        Initial vector difference between front and back neighboring particle
        positions.

    Returns
    -------
    jacobian : ArrayNx3x3
        The flow map Jacobian (deformation gradient) tensor for each particle
        group in 3D space.

    Notes
    -----
    The Jacobian is computed component-wise as:

        J_ij = Δx_i / ΔX_j

    where i, j ∈ {x, y, z} and Δx, ΔX are the deformed and initial separation
    vectors between neighboring particles.
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
    """
    Wrapper to compute the 2D flow map Jacobian (deformation gradient)
    from a `NeighboringParticles` instance.

    Parameters
    ----------
    particles : NeighboringParticles
        Dataclass containing particle positions and their corresponding
        neighbor displacements (current and initial) in 2D.

    Returns
    -------
    jacobian : ArrayNx2x2
        The 2D flow map Jacobian for each particle group.
    """
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
    """
    Wrapper to compute the 3D flow map Jacobian (deformation gradient)
    from a `NeighboringParticles` instance.

    Parameters
    ----------
    particles : NeighboringParticles
        Dataclass containing particle positions and their corresponding
        neighbor displacements (current and initial) in 3D.

    Returns
    -------
    jacobian : ArrayNx3x3
        The 3D flow map Jacobian for each particle group.
    """
    return _compute_flow_map_jacobian_in_numba_3x3(
        particles.positions,
        particles.delta_right_left,
        particles.initial_delta_right_left,
        particles.delta_top_bottom,
        particles.initial_delta_top_bottom,
        particles.delta_front_back,
        particles.initial_delta_front_back,
    )
