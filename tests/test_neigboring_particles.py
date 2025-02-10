import numpy as np
import pytest

from src.dtos import NeighboringParticles


@pytest.fixture
def sample_particles():
    """Creates a sample NeighboringParticles object for testing."""
    positions = np.array(
        [
            [0.0, 0.0],  # Left neighbor
            [1.0, 0.0],  # Right neighbor
            [0.5, 1.0],  # Top neighbor
            [0.5, 0.0],  # Bottom neighbor
        ],
        dtype=np.float32,
    )
    return NeighboringParticles(positions=positions)


def test_len(sample_particles):
    """Tests that the len function correctly returns the number of particle groups."""
    assert len(sample_particles) == 1  # Since there is 1 particle group (N=1)


def test_initial_deltas(sample_particles):
    """Tests if the initial deltas are correctly computed."""
    left, right = sample_particles.positions[0], sample_particles.positions[1]
    top, bottom = sample_particles.positions[2], sample_particles.positions[3]

    np.testing.assert_array_equal(
        sample_particles.initial_delta_top_bottom, (top - bottom).reshape(1, 2)
    )

    np.testing.assert_array_equal(
        sample_particles.initial_delta_right_left, (right - left).reshape(1, 2)
    )


def test_dynamic_delta_properties(sample_particles):
    """Tests if delta properties dynamically reflect updated values."""
    # Initial checks
    left, right = sample_particles.positions[0], sample_particles.positions[1]
    top, bottom = sample_particles.positions[2], sample_particles.positions[3]

    np.testing.assert_array_equal(
        sample_particles.delta_top_bottom, (top - bottom).reshape(1, 2)
    )
    np.testing.assert_array_equal(
        sample_particles.delta_right_left, (right - left).reshape(1, 2)
    )

    # Modify positions and check updated deltas
    sample_particles.positions[2] = np.array([0.6, 1.1], dtype=np.float32)  # Top
    sample_particles.positions[3] = np.array([0.4, 0.1], dtype=np.float32)  # Bottom
    np.testing.assert_array_equal(
        sample_particles.delta_top_bottom,
        (sample_particles.positions[2] - sample_particles.positions[3]).reshape(1, 2),
    )

    sample_particles.positions[0] = np.array([0.1, 0.1], dtype=np.float32)  # Left
    sample_particles.positions[1] = np.array([1.1, 0.1], dtype=np.float32)  # Right
    np.testing.assert_array_equal(
        sample_particles.delta_right_left,
        (sample_particles.positions[1] - sample_particles.positions[0]).reshape(1, 2),
    )


def test_independence_of_initial_deltas(sample_particles):
    """Tests that initial deltas remain unchanged after modifying attributes."""
    original_initial_delta_top_bottom = sample_particles.initial_delta_top_bottom.copy()
    original_initial_delta_right_left = sample_particles.initial_delta_right_left.copy()

    # Modify positions
    sample_particles.positions[2] = np.array([0.8, 1.2], dtype=np.float32)  # Top
    sample_particles.positions[3] = np.array([0.6, 0.2], dtype=np.float32)  # Bottom

    # Ensure initial deltas remain unchanged
    np.testing.assert_array_equal(
        sample_particles.initial_delta_top_bottom, original_initial_delta_top_bottom
    )
    np.testing.assert_array_equal(
        sample_particles.initial_delta_right_left, original_initial_delta_right_left
    )


def test_initial_centroid(sample_particles):
    """Tests if the initial centroid is correctly computed."""
    expected_centroid = np.mean(sample_particles.positions, axis=0).reshape(1, 2)
    np.testing.assert_array_equal(sample_particles.initial_centroid, expected_centroid)


def test_dynamic_centroid(sample_particles):
    """Tests if centroid property dynamically reflects updated values."""
    expected_centroid = np.mean(sample_particles.positions, axis=0).reshape(1, 2)
    np.testing.assert_array_equal(sample_particles.centroid, expected_centroid)

    # Modify positions and check updated centroid
    sample_particles.positions[0] = np.array([0.2, 0.2], dtype=np.float32)  # Left
    sample_particles.positions[1] = np.array([1.2, 0.2], dtype=np.float32)  # Right
    sample_particles.positions[2] = np.array([0.7, 1.3], dtype=np.float32)  # Top
    sample_particles.positions[3] = np.array([0.7, 0.1], dtype=np.float32)  # Bottom

    expected_centroid = np.mean(sample_particles.positions, axis=0).reshape(1, 2)
    np.testing.assert_array_equal(sample_particles.centroid, expected_centroid)
