import numpy as np
import pytest

from src.dtos import NeighboringParticles


@pytest.fixture
def sample_particles():
    """Creates a sample NeighboringParticles object for testing."""
    return NeighboringParticles(
        left=np.array([[0.0, 0.0]], dtype=np.float32),
        right=np.array([[1.0, 0.0]], dtype=np.float32),
        top=np.array([[0.5, 1.0]], dtype=np.float32),
        bottom=np.array([[0.5, 0.0]], dtype=np.float32),
    )


def test_initial_deltas(sample_particles):
    """Tests if the initial deltas are correctly computed."""
    np.testing.assert_array_equal(
        sample_particles.initial_delta_top_bottom,
        sample_particles.top - sample_particles.bottom,
    )
    np.testing.assert_array_equal(
        sample_particles.initial_delta_right_left,
        sample_particles.right - sample_particles.left,
    )


def test_dynamic_delta_properties(sample_particles):
    """Tests if delta properties dynamically reflect updated values."""
    # Initial checks
    assert np.all(
        sample_particles.delta_top_bottom
        == sample_particles.top - sample_particles.bottom
    )
    assert np.all(
        sample_particles.delta_right_left
        == sample_particles.right - sample_particles.left
    )

    # Update attributes and check if properties reflect the changes
    sample_particles.top = np.array([[0.6, 1.1]], dtype=np.float32)
    sample_particles.bottom = np.array([[0.4, 0.1]], dtype=np.float32)

    expected_delta_top_bottom = sample_particles.top - sample_particles.bottom
    np.testing.assert_array_equal(
        sample_particles.delta_top_bottom, expected_delta_top_bottom
    )

    sample_particles.left = np.array([[0.1, 0.1]], dtype=np.float32)
    sample_particles.right = np.array([[1.1, 0.1]], dtype=np.float32)

    expected_delta_right_left = sample_particles.right - sample_particles.left
    np.testing.assert_array_equal(
        sample_particles.delta_right_left, expected_delta_right_left
    )


def test_independence_of_initial_deltas(sample_particles):
    """Tests that initial deltas remain unchanged after modifying attributes."""
    original_initial_delta_top_bottom = sample_particles.initial_delta_top_bottom.copy()
    original_initial_delta_right_left = sample_particles.initial_delta_right_left.copy()

    # Modify attributes
    sample_particles.top = np.array([[0.8, 1.2]], dtype=np.float32)
    sample_particles.bottom = np.array([[0.6, 0.2]], dtype=np.float32)

    # Ensure initial deltas remain unchanged
    np.testing.assert_array_equal(
        sample_particles.initial_delta_top_bottom, original_initial_delta_top_bottom
    )
    np.testing.assert_array_equal(
        sample_particles.initial_delta_right_left, original_initial_delta_right_left
    )
