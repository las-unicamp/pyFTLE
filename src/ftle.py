# ruff: noqa: N806

import numpy as np
from numba import njit  # type: ignore

from src.my_types import ArrayFloat64N, ArrayFloat64Nx2x2


@njit
def compute_cauchy_green(flow_map_jacobian: ArrayFloat64Nx2x2) -> ArrayFloat64Nx2x2:
    """Compute the Cauchy-Green deformation tensor for each Jacobian."""
    num_particles = flow_map_jacobian.shape[0]
    cauchy_green_tensor = np.empty((num_particles, 2, 2))

    for i in range(num_particles):
        F = flow_map_jacobian[i]
        cauchy_green_tensor[i, 0, 0] = F[0, 0] * F[0, 0] + F[1, 0] * F[1, 0]  # A
        cauchy_green_tensor[i, 0, 1] = F[0, 0] * F[0, 1] + F[1, 0] * F[1, 1]  # B
        cauchy_green_tensor[i, 1, 0] = F[0, 1] * F[0, 0] + F[1, 1] * F[1, 0]  # C
        cauchy_green_tensor[i, 1, 1] = F[0, 1] * F[0, 1] + F[1, 1] * F[1, 1]  # D

    return cauchy_green_tensor


@njit
def max_eigenvalue_2x2(cauchy_green_tensor: ArrayFloat64Nx2x2) -> ArrayFloat64N:
    """Compute the maximum eigenvalue for each 2x2 Cauchy-Green tensor."""
    num_particles = cauchy_green_tensor.shape[0]
    max_eigvals = np.empty(num_particles)

    for i in range(num_particles):
        A = cauchy_green_tensor[i, 0, 0]
        B = cauchy_green_tensor[i, 0, 1]
        C = cauchy_green_tensor[i, 1, 0]
        D = cauchy_green_tensor[i, 1, 1]

        # Compute eigenvalues of a 2x2 matrix analytically
        trace = A + D
        determinant = A * D - B * C
        lambda1 = 0.5 * (trace + np.sqrt(trace**2 - 4 * determinant))
        lambda2 = 0.5 * (trace - np.sqrt(trace**2 - 4 * determinant))

        max_eigvals[i] = max(lambda1, lambda2)

    return max_eigvals


@njit
def compute_ftle(
    flow_map_jacobian: ArrayFloat64Nx2x2, map_period: float
) -> ArrayFloat64N:
    """Compute FTLE using a Numba-optimized approach."""
    cauchy_green_tensor = compute_cauchy_green(flow_map_jacobian)
    max_eigvals = max_eigenvalue_2x2(cauchy_green_tensor)
    return (1 / map_period) * np.log(np.sqrt(max_eigvals))
