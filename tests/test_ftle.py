# ruff: noqa: N806, N802
import numpy as np

from pyftle.ftle import (
    compute_cauchy_green_2x2,
    compute_cauchy_green_3x3,
    compute_ftle_2x2,
    compute_ftle_3x3,
    max_eigenvalue_2x2,
    max_eigenvalue_3x3,
)


# -------------------------
# compute_cauchy_green tests
# -------------------------
def test_compute_cauchy_green_2x2_matches_FT_FTt(jacobians_2x2):
    computed = compute_cauchy_green_2x2(jacobians_2x2)

    expected = np.einsum("...ij,...kj->...ik", jacobians_2x2, jacobians_2x2)
    np.testing.assert_allclose(computed, expected, rtol=1e-12)


def test_compute_cauchy_green_3x3_matches_FT_FTt(jacobians_3x3):
    computed = compute_cauchy_green_3x3(jacobians_3x3)

    expected = jacobians_3x3 @ np.transpose(jacobians_3x3, (0, 2, 1))
    np.testing.assert_allclose(computed, expected, rtol=1e-12)


# -------------------------
# max_eigenvalue tests
# -------------------------
def test_max_eigenvalue_2x2_matches_numpy(jacobians_2x2):
    cg = compute_cauchy_green_2x2(jacobians_2x2)

    computed = max_eigenvalue_2x2(cg)
    expected = np.max(np.linalg.eigvals(cg), axis=1)
    np.testing.assert_allclose(computed, expected, rtol=1e-10)


def test_max_eigenvalue_3x3_matches_numpy(jacobians_3x3):
    cg = compute_cauchy_green_3x3(jacobians_3x3)

    computed = max_eigenvalue_3x3(cg)
    expected = np.max(np.linalg.eigvals(cg), axis=1)
    np.testing.assert_allclose(computed, expected, rtol=1e-10)


# -------------------------
# compute_ftle tests
# -------------------------
def test_compute_ftle_2x2_matches_manual(jacobians_2x2):
    map_period = 1.0

    cauchy_green_tensor = np.einsum("...ij,...kj->...ik", jacobians_2x2, jacobians_2x2)
    max_eigvals = np.max(np.linalg.eigvals(cauchy_green_tensor), axis=1)
    expected_ftle = (1 / map_period) * np.log(np.sqrt(max_eigvals))

    computed_ftle = compute_ftle_2x2(jacobians_2x2, map_period)
    np.testing.assert_allclose(computed_ftle, expected_ftle, rtol=1e-8)


def test_compute_ftle_3x3_matches_manual(jacobians_3x3):
    map_period = 2.0

    cauchy_green_tensor = jacobians_3x3 @ np.transpose(jacobians_3x3, (0, 2, 1))
    max_eigvals = np.max(np.linalg.eigvals(cauchy_green_tensor), axis=1)
    expected_ftle = (1 / map_period) * np.log(np.sqrt(max_eigvals))

    computed_ftle = compute_ftle_3x3(jacobians_3x3, map_period)
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
