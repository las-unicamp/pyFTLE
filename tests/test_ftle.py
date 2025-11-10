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


def test_compute_cauchy_green_2x2_matches_FT_FTt(generate_2x2_jacobians):
    """Tests that compute_cauchy_green_2x2 correctly computes the
    Cauchy-Green tensor.

    Args:
        generate_2x2_jacobians (np.ndarray): Fixture providing 2x2 Jacobian matrices.

    Flow:
        generate_2x2_jacobians -> compute_cauchy_green_2x2 -> computed
        computed == expected (calculated manually)
    """
    jacobians = generate_2x2_jacobians
    computed = compute_cauchy_green_2x2(jacobians)

    expected = np.einsum("...ij,...kj->...ik", jacobians, jacobians)
    np.testing.assert_allclose(computed, expected, rtol=1e-12)


def test_compute_cauchy_green_3x3_matches_FT_FTt(generate_3x3_jacobians):
    """Tests that compute_cauchy_green_3x3 correctly computes the
    Cauchy-Green tensor.

    Args:
        generate_3x3_jacobians (np.ndarray): Fixture providing 3x3 Jacobian matrices.

    Flow:
        generate_3x3_jacobians -> compute_cauchy_green_3x3 -> computed
        computed == expected (calculated manually)
    """
    jacobians = generate_3x3_jacobians
    computed = compute_cauchy_green_3x3(jacobians)

    expected = jacobians @ np.transpose(jacobians, (0, 2, 1))
    np.testing.assert_allclose(computed, expected, rtol=1e-12)


def test_max_eigenvalue_2x2_matches_numpy(generate_2x2_jacobians):
    """Tests that max_eigenvalue_2x2 returns the correct maximum eigenvalue.

    Args:
        generate_2x2_jacobians (np.ndarray): Fixture providing 2x2 Jacobian matrices.

    Flow:
        generate_2x2_jacobians -> compute_cauchy_green_2x2 -> cg
        cg -> max_eigenvalue_2x2 -> computed
        computed == expected (calculated using numpy's eigvals)
    """
    jacobians = generate_2x2_jacobians
    cg = compute_cauchy_green_2x2(jacobians)

    computed = max_eigenvalue_2x2(cg)
    expected = np.max(np.linalg.eigvals(cg), axis=1)
    np.testing.assert_allclose(computed, expected, rtol=1e-10)


def test_max_eigenvalue_3x3_matches_numpy(generate_3x3_jacobians):
    """Tests that max_eigenvalue_3x3 returns the correct maximum eigenvalue.

    Args:
        generate_3x3_jacobians (np.ndarray): Fixture providing 3x3 Jacobian matrices.

    Flow:
        generate_3x3_jacobians -> compute_cauchy_green_3x3 -> cg
        cg -> max_eigenvalue_3x3 -> computed
        computed == expected (calculated using numpy's eigvals)
    """
    jacobians = generate_3x3_jacobians
    cg = compute_cauchy_green_3x3(jacobians)

    computed = max_eigenvalue_3x3(cg)
    expected = np.max(np.linalg.eigvals(cg), axis=1)
    np.testing.assert_allclose(computed, expected, rtol=1e-10)


def test_compute_ftle_2x2_matches_manual(generate_2x2_jacobians):
    """Tests that compute_ftle_2x2 correctly computes the FTLE field for
    2x2 Jacobians.

    Args:
        generate_2x2_jacobians (np.ndarray): Fixture providing 2x2 Jacobian matrices.

    Flow:
        generate_2x2_jacobians, map_period -> compute_ftle_2x2 -> computed_ftle
        computed_ftle == expected_ftle (calculated manually)
    """
    flow_map_jacobian = generate_2x2_jacobians
    map_period = 1.0

    cauchy_green_tensor = np.einsum(
        "...ij,...kj->...ik", flow_map_jacobian, flow_map_jacobian
    )
    max_eigvals = np.max(np.linalg.eigvals(cauchy_green_tensor), axis=1)
    expected_ftle = (1 / map_period) * np.log(np.sqrt(max_eigvals))

    computed_ftle = compute_ftle_2x2(flow_map_jacobian, map_period)
    np.testing.assert_allclose(computed_ftle, expected_ftle, rtol=1e-8)


def test_compute_ftle_3x3_matches_manual(generate_3x3_jacobians):
    """Tests that compute_ftle_3x3 correctly computes the FTLE field for
    3x3 Jacobians.

    Args:
        generate_3x3_jacobians (np.ndarray): Fixture providing 3x3 Jacobian matrices.

    Flow:
        generate_3x3_jacobians, map_period -> compute_ftle_3x3 -> computed_ftle
        computed_ftle == expected_ftle (calculated manually)
    """
    flow_map_jacobian = generate_3x3_jacobians
    map_period = 2.0

    cauchy_green_tensor = flow_map_jacobian @ np.transpose(flow_map_jacobian, (0, 2, 1))
    max_eigvals = np.max(np.linalg.eigvals(cauchy_green_tensor), axis=1)
    expected_ftle = (1 / map_period) * np.log(np.sqrt(max_eigvals))

    computed_ftle = compute_ftle_3x3(flow_map_jacobian, map_period)
    np.testing.assert_allclose(computed_ftle, expected_ftle, rtol=1e-8)


def test_zero_deformation_returns_zero_ftle():
    """Tests that zero deformation results in zero FTLE.

    Flow:
        Identity Jacobian -> compute_ftle_2x2 -> ftle
        ftle == array of zeros
    """
    F = np.repeat(np.eye(2)[None, :, :], 3, axis=0)
    ftle = compute_ftle_2x2(F, map_period=1.0)
    np.testing.assert_allclose(ftle, np.zeros(3), atol=1e-12)


def test_large_deformation_increases_ftle():
    """Tests that larger deformation increases FTLE magnitude.

    Flow:
        Small deformation Jacobian -> compute_ftle_2x2 -> ftle_small
        Large deformation Jacobian -> compute_ftle_2x2 -> ftle_large
        ftle_large > ftle_small
    """
    F_small = np.repeat(np.eye(2)[None, :, :], 1, axis=0)
    F_large = np.repeat((2.0 * np.eye(2))[None, :, :], 1, axis=0)

    ftle_small = compute_ftle_2x2(F_small, map_period=1.0)
    ftle_large = compute_ftle_2x2(F_large, map_period=1.0)
    assert ftle_large > ftle_small
