import numpy as np
import pytest

from pyftle.my_types import Array4Nx2, Array6Nx3
from pyftle.particles import NeighboringParticles


@pytest.fixture(params=["2D", "3D"])
def sample_particles(request):
    """Creates a NeighboringParticles object for 2D or 3D test cases."""
    if request.param == "2D":
        positions: Array4Nx2 = np.array(
            [
                [0.0, 0.0],  # Left neighbor
                [1.0, 0.0],  # Right neighbor
                [0.5, 1.0],  # Top neighbor
                [0.5, 0.0],  # Bottom neighbor
            ],
            dtype=Array4Nx2,
        )
    else:  # 3D
        positions: Array6Nx3 = np.array(
            [
                [0.0, 0.0, 0.0],  # Left
                [1.0, 0.0, 0.0],  # Right
                [0.5, 1.0, 0.0],  # Top
                [0.5, -1.0, 0.0],  # Bottom
                [0.5, 0.0, 1.0],  # Front
                [0.5, 0.0, -1.0],  # Back
            ],
            dtype=Array6Nx3,
        )
    return NeighboringParticles(positions=positions)


def test_len(sample_particles):
    """Ensure len() correctly returns number of particle groups."""
    assert len(sample_particles) == 1


def test_initial_deltas(sample_particles):
    """Check initial delta vectors are correctly computed."""
    dim = sample_particles.positions.shape[1]

    if dim == 2:
        left, right, top, bottom = sample_particles.positions
        np.testing.assert_array_equal(
            sample_particles.initial_delta_top_bottom, (top - bottom).reshape(1, dim)
        )
        np.testing.assert_array_equal(
            sample_particles.initial_delta_right_left, (right - left).reshape(1, dim)
        )
    else:
        left, right, top, bottom, front, back = sample_particles.positions
        np.testing.assert_array_equal(
            sample_particles.initial_delta_top_bottom, (top - bottom).reshape(1, dim)
        )
        np.testing.assert_array_equal(
            sample_particles.initial_delta_right_left, (right - left).reshape(1, dim)
        )
        np.testing.assert_array_equal(
            sample_particles.initial_delta_front_back, (front - back).reshape(1, dim)
        )


def test_dynamic_delta_properties(sample_particles):
    """Ensure delta properties dynamically update with positions."""
    dim = sample_particles.positions.shape[1]

    if dim == 2:
        left, right, top, bottom = sample_particles.positions
        np.testing.assert_array_equal(
            sample_particles.delta_top_bottom, (top - bottom).reshape(1, dim)
        )
        np.testing.assert_array_equal(
            sample_particles.delta_right_left, (right - left).reshape(1, dim)
        )

        # Modify top/bottom and right/left
        sample_particles.positions[2] = np.array([0.6, 1.1])
        sample_particles.positions[3] = np.array([0.4, 0.1])
        np.testing.assert_array_equal(
            sample_particles.delta_top_bottom,
            (sample_particles.positions[2] - sample_particles.positions[3]).reshape(
                1, dim
            ),
        )

        sample_particles.positions[0] = np.array([0.1, 0.1])
        sample_particles.positions[1] = np.array([1.1, 0.1])
        np.testing.assert_array_equal(
            sample_particles.delta_right_left,
            (sample_particles.positions[1] - sample_particles.positions[0]).reshape(
                1, dim
            ),
        )

    else:
        left, right, top, bottom, front, back = sample_particles.positions
        np.testing.assert_array_equal(
            sample_particles.delta_top_bottom, (top - bottom).reshape(1, dim)
        )
        np.testing.assert_array_equal(
            sample_particles.delta_right_left, (right - left).reshape(1, dim)
        )
        np.testing.assert_array_equal(
            sample_particles.delta_front_back, (front - back).reshape(1, dim)
        )

        # Modify front/back and recheck
        sample_particles.positions[4] = np.array([0.6, 0.0, 1.1])
        sample_particles.positions[5] = np.array([0.4, 0.0, -1.1])
        np.testing.assert_array_equal(
            sample_particles.delta_front_back,
            (sample_particles.positions[4] - sample_particles.positions[5]).reshape(
                1, dim
            ),
        )


def test_independence_of_initial_deltas(sample_particles):
    """Ensure initial deltas remain constant after modifying positions."""
    original_tb = sample_particles.initial_delta_top_bottom.copy()
    original_rl = sample_particles.initial_delta_right_left.copy()
    original_fb = None  # avoid unbound variable warning

    if sample_particles.positions.shape[1] == 3:
        original_fb = sample_particles.initial_delta_front_back.copy()

    # Modify positions (simulate motion)
    sample_particles.positions += 0.2

    np.testing.assert_array_equal(
        sample_particles.initial_delta_top_bottom, original_tb
    )
    np.testing.assert_array_equal(
        sample_particles.initial_delta_right_left, original_rl
    )

    if original_fb is not None:
        np.testing.assert_array_equal(
            sample_particles.initial_delta_front_back, original_fb
        )


def test_initial_centroid(sample_particles):
    """Check initial centroid computation."""
    expected = np.mean(sample_particles.positions, axis=0).reshape(1, -1)
    np.testing.assert_array_equal(sample_particles.initial_centroid, expected)


def test_dynamic_centroid(sample_particles):
    """Ensure centroid updates dynamically with positions."""
    expected = np.mean(sample_particles.positions, axis=0).reshape(1, -1)
    np.testing.assert_array_equal(sample_particles.centroid, expected)

    # Modify positions and verify centroid updates
    sample_particles.positions += np.random.uniform(
        -0.1, 0.1, sample_particles.positions.shape
    )
    expected = np.mean(sample_particles.positions, axis=0).reshape(1, -1)
    np.testing.assert_array_equal(sample_particles.centroid, expected)
