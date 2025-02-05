import numpy as np

from src.dtos import NeighboringParticles


def test_neighboring_particles():
    # Create sample data
    left = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    right = np.array([[1.0, 0.0], [2.0, 1.0]], dtype=np.float32)
    top = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    bottom = np.array([[0.5, 0.0], [1.5, 1.0]], dtype=np.float32)

    particles = NeighboringParticles(left=left, right=right, top=top, bottom=bottom)

    expected_delta_top_bottom = top - bottom
    expected_delta_right_left = right - left

    np.testing.assert_array_equal(particles.delta_top_bottom, expected_delta_top_bottom)
    np.testing.assert_array_equal(particles.delta_right_left, expected_delta_right_left)
