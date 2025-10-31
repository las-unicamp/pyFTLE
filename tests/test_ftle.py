# ruff: noqa: N806, N802
import numpy as np
import pytest

from pyftle.ftle import (
    compute_cauchy_green_2x2,
    compute_cauchy_green_3x3,
    compute_ftle_2x2,
    compute_ftle_3x3,
    max_eigenvalue_2x2,
    max_eigenvalue_3x3,
)
from pyftle.my_types import ArrayNx2x2, ArrayNx3x3


# -------------------------
# Helpers
# -------------------------
def generate_2x2_jacobians() -> ArrayNx2x2:
    """Return a small array of 2×2 Jacobians for testing."""
    return np.array(
        [
            [[2.0, 0.0], [0.0, 1.0]],
            [[1.5, 0.5], [0.5, 1.5]],
        ],
        dtype=np.float64,
    )


def generate_3x3_jacobians() -> ArrayNx3x3:
    """Return a small array of 3×3 Jacobians for testing."""
    return np.array(
        [
            np.eye(3),
            [[1.0, 0.2, 0.0], [0.0, 1.5, 0.1], [0.0, 0.0, 1.2]],
        ],
        dtype=np.float64,
    )


# -------------------------
# compute_cauchy_green tests
# -------------------------
def test_compute_cauchy_green_2x2_matches_FT_FTt():
    jacobians = generate_2x2_jacobians()
    computed = compute_cauchy_green_2x2(jacobians)

    expected = np.einsum("...ij,...kj->...ik", jacobians, jacobians)
    np.testing.assert_allclose(computed, expected, rtol=1e-12)


def test_compute_cauchy_green_3x3_matches_FT_FTt():
    jacobians = generate_3x3_jacobians()
    computed = compute_cauchy_green_3x3(jacobians)

    expected = jacobians @ np.transpose(jacobians, (0, 2, 1))
    np.testing.assert_allclose(computed, expected, rtol=1e-12)


# -------------------------
# max_eigenvalue tests
# -------------------------
def test_max_eigenvalue_2x2_matches_numpy():
    jacobians = generate_2x2_jacobians()
    cg = compute_cauchy_green_2x2(jacobians)

    computed = max_eigenvalue_2x2(cg)
    expected = np.max(np.linalg.eigvals(cg), axis=1)
    np.testing.assert_allclose(computed, expected, rtol=1e-10)


def test_max_eigenvalue_3x3_matches_numpy():
    jacobians = generate_3x3_jacobians()
    cg = compute_cauchy_green_3x3(jacobians)

    computed = max_eigenvalue_3x3(cg)
    expected = np.max(np.linalg.eigvals(cg), axis=1)
    np.testing.assert_allclose(computed, expected, rtol=1e-10)


# -------------------------
# compute_ftle tests
# -------------------------
def test_compute_ftle_2x2_matches_manual():
    flow_map_jacobian = generate_2x2_jacobians()
    map_period = 1.0

    cauchy_green_tensor = np.einsum(
        "...ij,...kj->...ik", flow_map_jacobian, flow_map_jacobian
    )
    max_eigvals = np.max(np.linalg.eigvals(cauchy_green_tensor), axis=1)
    expected_ftle = (1 / map_period) * np.log(np.sqrt(max_eigvals))

    computed_ftle = compute_ftle_2x2(flow_map_jacobian, map_period)
    np.testing.assert_allclose(computed_ftle, expected_ftle, rtol=1e-8)


def test_compute_ftle_3x3_matches_manual():
    flow_map_jacobian = generate_3x3_jacobians()
    map_period = 2.0

    cauchy_green_tensor = flow_map_jacobian @ np.transpose(flow_map_jacobian, (0, 2, 1))
    max_eigvals = np.max(np.linalg.eigvals(cauchy_green_tensor), axis=1)
    expected_ftle = (1 / map_period) * np.log(np.sqrt(max_eigvals))

    computed_ftle = compute_ftle_3x3(flow_map_jacobian, map_period)
    np.testing.assert_allclose(computed_ftle, expected_ftle, rtol=1e-8)


# -------------------------
# Edge and numerical stability tests
# -------------------------
def test_zero_deformation_returns_zero_ftle():
    """If flow map Jacobian is identity, FTLE must be zero."""
    F = np.repeat(np.eye(2)[None, :, :], 3, axis=0)
    ftle = compute_ftle_2x2(F, map_period=1.0)
    np.testing.assert_allclose(ftle, np.zeros(3), atol=1e-12)


def test_large_deformation_increases_ftle():
    """Larger stretching should increase FTLE magnitude."""
    F_small = np.repeat(np.eye(2)[None, :, :], 1, axis=0)
    F_large = np.repeat((2.0 * np.eye(2))[None, :, :], 1, axis=0)

    ftle_small = compute_ftle_2x2(F_small, map_period=1.0)
    ftle_large = compute_ftle_2x2(F_large, map_period=1.0)
    assert ftle_large > ftle_small


if __name__ == "__main__":
    pytest.main()
