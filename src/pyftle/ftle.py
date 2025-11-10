# ruff: noqa: N806

import numpy as np
from numba import njit  # type: ignore

from pyftle.my_types import ArrayN, ArrayNx2x2, ArrayNx3x3


@njit
def compute_cauchy_green_2x2(flow_map_jacobian: ArrayNx2x2) -> ArrayNx2x2:
    r"""
    Compute the 2D right Cauchy-Green deformation tensor for each flow map Jacobian.

    The right Cauchy-Green tensor is defined as:

    .. math::

        \mathbf{C} = \mathbf{F}^\mathsf{T} \mathbf{F},

    where :math:`\mathbf{F}` is the deformation gradient (flow map Jacobian).

    This Numba-optimized implementation explicitly expands the matrix product
    for performance in 2D.

    Parameters
    ----------
    flow_map_jacobian : ArrayNx2x2
        Deformation gradient tensors (Jacobian matrices) for all particle
        groups. Shape ``(N, 2, 2)``.

    Returns
    -------
    cauchy_green_tensor : ArrayNx2x2
        The right Cauchy-Green deformation tensors. Shape ``(N, 2, 2)``.

    Notes
    -----

    For each tensor :math:`\mathbf{F}`, defined as

    .. math::

        \mathbf{F} =
        \begin{bmatrix}
        F_{11} & F_{12} \\
        F_{21} & F_{22}
        \end{bmatrix},

    the resulting :math:`\mathbf{C}` is:

    .. math::

        \mathbf{C} =
        \begin{bmatrix}
        F_{11}^2 + F_{21}^2 & F_{11} F_{12} + F_{21} F_{22} \\
        F_{12} F_{11} + F_{22} F_{21} & F_{12}^2 + F_{22}^2
        \end{bmatrix}.
    """
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
    r"""
    Compute the 3D right Cauchy-Green deformation tensor for each flow map Jacobian.

    The tensor is given by:

    .. math::

        \mathbf{C} = \mathbf{F}^\mathsf{T} \mathbf{F},

    where :math:`\mathbf{F}` is the deformation gradient.

    Parameters
    ----------
    flow_map_jacobian : ArrayNx3x3
        Deformation gradient tensors (Jacobian matrices) for all particle
        groups. Shape ``(N, 3, 3)``.

    Returns
    -------
    cauchy_green_tensor : ArrayNx3x3
        The right Cauchy-Green deformation tensors. Shape ``(N, 3, 3)``.
    """

    return flow_map_jacobian @ np.transpose(flow_map_jacobian, (0, 2, 1))


def max_eigenvalue_3x3(cauchy_green_tensor: ArrayNx3x3) -> ArrayN:
    r"""
    Compute the maximum eigenvalue of the 3D right Cauchy-Green deformation tensor.

    Parameters
    ----------
    cauchy_green_tensor : ArrayNx3x3
        Right Cauchy-Green deformation tensors. Shape ``(N, 3, 3)``.

    Returns
    -------
    max_eigvals : ArrayN
        Maximum eigenvalue of each tensor. Shape ``(N,)``.

    Notes
    -----
    The eigenvalues correspond to the squared principal stretch ratios.
    """

    eigenvalues = np.linalg.eigvals(cauchy_green_tensor)
    return np.max(eigenvalues, axis=1)


@njit
def max_eigenvalue_2x2(cauchy_green_tensor: ArrayNx2x2) -> ArrayN:
    r"""
    Compute the maximum eigenvalue of each 2D right Cauchy-Green deformation tensor.

    Parameters
    ----------
    cauchy_green_tensor : ArrayNx2x2
        Right Cauchy-Green deformation tensors. Shape ``(N, 2, 2)``.

    Returns
    -------
    max_eigvals : ArrayN
        Maximum eigenvalue of each tensor. Shape ``(N,)``.

    Notes
    -----
    The eigenvalues are computed analytically from the characteristic equation:

    .. math::

        \lambda = \frac{\operatorname{tr}(\mathbf{C}) \pm
        \sqrt{\operatorname{tr}(\mathbf{C})^2 - 4 \det(\mathbf{C})}}{2}.
    """
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
    r"""
    Compute the Finite-Time Lyapunov Exponent (FTLE) field in 2D.

    The FTLE quantifies the rate of separation between neighboring trajectories
    over a finite time interval and is given by:

    .. math::

        \sigma = \frac{1}{|T|}
        \ln \sqrt{\lambda_{\max}(\mathbf{C})},

    where :math:`\lambda_{\max}` is the maximum eigenvalue of the right
    Cauchy-Green tensor :math:`\mathbf{C} = \mathbf{F}^\mathsf{T} \mathbf{F}`.

    Parameters
    ----------
    flow_map_jacobian : ArrayNx2x2
        Deformation gradient (flow map Jacobian) for each particle group.
        Shape ``(N, 2, 2)``.
    map_period : float
        Finite integration time :math:`T` (positive for forward time, negative
        for backward time).

    Returns
    -------
    ftle : ArrayN
        Finite-Time Lyapunov Exponent for each particle. Shape ``(N,)``.
    """
    cauchy_green_tensor = compute_cauchy_green_2x2(flow_map_jacobian)
    max_eigvals = max_eigenvalue_2x2(cauchy_green_tensor)
    return (1 / map_period) * np.log(np.sqrt(max_eigvals))


def compute_ftle_3x3(flow_map_jacobian: ArrayNx3x3, map_period: float) -> ArrayN:
    r"""
    Compute the Finite-Time Lyapunov Exponent (FTLE) field in 3D.

    The FTLE quantifies the rate of exponential separation of trajectories over
    a finite time interval, based on the largest eigenvalue of the
    Cauchy-Green deformation tensor.

    .. math::

        \sigma = \frac{1}{|T|}
        \ln \sqrt{\lambda_{\max}(\mathbf{C})},

    where :math:`\mathbf{C} = \mathbf{F}^\mathsf{T} \mathbf{F}`.

    Parameters
    ----------
    flow_map_jacobian : ArrayNx3x3
        Deformation gradient (flow map Jacobian) for each particle group.
        Shape ``(N, 3, 3)``.
    map_period : float
        Finite integration time :math:`T` (positive for forward time, negative
        for backward time).

    Returns
    -------
    ftle : ArrayN
        Finite-Time Lyapunov Exponent for each particle. Shape ``(N,)``.
    """
    cauchy_green_tensor = compute_cauchy_green_3x3(flow_map_jacobian)
    max_eigvals = max_eigenvalue_3x3(cauchy_green_tensor)

    return (1 / map_period) * np.log(np.sqrt(max_eigvals))
