# ruff: noqa: N806

import numpy as np
from numba import njit  # type: ignore

from pyftle.my_types import ArrayN, ArrayNx2x2, ArrayNx3x3


@njit
def compute_cauchy_green_2x2(flow_map_jacobian: ArrayNx2x2) -> ArrayNx2x2:
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


def compute_cauchy_green_3x3(flow_map_jacobian: ArrayNx3x3) -> ArrayNx3x3:
    """Compute the Cauchy-Green deformation tensor for each Jacobian."""

    return flow_map_jacobian @ np.transpose(flow_map_jacobian, (0, 2, 1))


def max_eigenvalue_3x3(cauchy_green_tensor: ArrayNx3x3) -> ArrayN:
    """Compute the Cauchy-Green deformation tensor for each Jacobian."""

    eigenvalues = np.linalg.eigvals(cauchy_green_tensor)
    return np.max(eigenvalues, axis=1)


@njit
def max_eigenvalue_2x2(cauchy_green_tensor: ArrayNx2x2) -> ArrayN:
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
def compute_ftle_2x2(flow_map_jacobian: ArrayNx2x2, map_period: float) -> ArrayN:
    """Compute FTLE using a Numba-optimized approach."""
    cauchy_green_tensor = compute_cauchy_green_2x2(flow_map_jacobian)
    max_eigvals = max_eigenvalue_2x2(cauchy_green_tensor)
    return (1 / map_period) * np.log(np.sqrt(max_eigvals))


def compute_ftle_3x3(flow_map_jacobian: ArrayNx3x3, map_period: float) -> ArrayN:
    """Compute FTLE using a Numba-optimized approach."""
    cauchy_green_tensor = compute_cauchy_green_3x3(flow_map_jacobian)
    max_eigvals = max_eigenvalue_3x3(cauchy_green_tensor)

    return (1 / map_period) * np.log(np.sqrt(max_eigvals))
