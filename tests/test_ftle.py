import numpy as np
import pytest

from src.ftle import compute_ftle


def test_compute_ftle():
    # Define a small test case
    flow_map_jacobian = np.array(
        [[[[2.0, 0.0], [0.0, 1.0]]], [[[1.5, 0.5], [0.5, 1.5]]]], dtype=np.float32
    ).reshape(2, 2, 2)

    map_period = 1.0  # Example period

    # Expected computation
    cauchy_green_tensor = np.einsum(
        "...ji,...jk->...ik", flow_map_jacobian, flow_map_jacobian
    )
    max_eigvals = np.linalg.eigvals(cauchy_green_tensor).max(axis=1)
    expected_ftle = 1 / map_period * np.log(np.sqrt(max_eigvals))

    # Compute FTLE
    computed_ftle = compute_ftle(flow_map_jacobian, map_period)

    # Assert values are close
    np.testing.assert_allclose(computed_ftle, expected_ftle, rtol=1e-6)


if __name__ == "__main__":
    pytest.main()
