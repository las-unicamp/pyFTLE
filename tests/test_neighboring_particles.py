import numpy as np


def test_len(sample_particles):
    """Tests that len() correctly returns the number of particle groups.

    Args:
        sample_particles (NeighboringParticles): Fixture providing sample
            particle data.

    Flow:
        len(sample_particles) -> result
        result == 1
    """
    assert len(sample_particles) == 1


def test_initial_deltas(sample_particles):
    """Tests that initial delta vectors are correctly computed.

    Args:
        sample_particles (NeighboringParticles): Fixture providing sample
            particle data.

    Flow:
        sample_particles.positions -> initial_delta_top_bottom,
        initial_delta_right_left, initial_delta_front_back
        Each initial_delta == expected (calculated manually)
    """
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
    """Tests that delta properties dynamically update with positions.

    Args:
        sample_particles (NeighboringParticles): Fixture providing sample
            particle data.

    Flow:
        sample_particles.positions -> delta_top_bottom, delta_right_left,
        delta_front_back
        Each delta == expected (calculated manually)
        Modify sample_particles.positions -> re-check deltas -> deltas update
        correctly
    """
    dim = sample_particles.positions.shape[1]

    if dim == 2:
        left, right, top, bottom = sample_particles.positions
        np.testing.assert_array_equal(
            sample_particles.delta_top_bottom, (top - bottom).reshape(1, dim)
        )
        np.testing.assert_array_equal(
            sample_particles.delta_right_left, (right - left).reshape(1, dim)
        )

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

        sample_particles.positions[4] = np.array([0.6, 0.0, 1.1])
        sample_particles.positions[5] = np.array([0.4, 0.0, -1.1])
        np.testing.assert_array_equal(
            sample_particles.delta_front_back,
            (sample_particles.positions[4] - sample_particles.positions[5]).reshape(
                1, dim
            ),
        )


def test_independence_of_initial_deltas(sample_particles):
    """Tests that initial deltas remain constant after modifying positions.

    Args:
        sample_particles (NeighboringParticles): Fixture providing sample
            particle data.

    Flow:
        Store initial deltas -> Modify sample_particles.positions ->
        initial_delta_top_bottom == original_tb
        initial_delta_right_left == original_rl
        initial_delta_front_back == original_fb (if 3D)
    """
    original_tb = sample_particles.initial_delta_top_bottom.copy()
    original_rl = sample_particles.initial_delta_right_left.copy()
    original_fb = None

    if sample_particles.positions.shape[1] == 3:
        original_fb = sample_particles.initial_delta_front_back.copy()

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
    """Tests initial centroid computation.

    Args:
        sample_particles (NeighboringParticles): Fixture providing sample
            particle data.

    Flow:
        sample_particles.positions -> initial_centroid
        initial_centroid == expected (calculated manually)
    """
    expected = np.mean(sample_particles.positions, axis=0).reshape(1, -1)
    np.testing.assert_array_equal(sample_particles.initial_centroid, expected)


def test_dynamic_centroid(sample_particles):
    """Tests that the centroid updates dynamically with positions.

    Args:
        sample_particles (NeighboringParticles): Fixture providing sample
            particle data.

    Flow:
        sample_particles.positions -> centroid
        centroid == expected (calculated manually)
        Modify sample_particles.positions -> re-check centroid -> centroid
        updates correctly
    """
    expected = np.mean(sample_particles.positions, axis=0).reshape(1, -1)
    np.testing.assert_array_equal(sample_particles.centroid, expected)

    sample_particles.positions += np.random.uniform(
        -0.1, 0.1, sample_particles.positions.shape
    )
    expected = np.mean(sample_particles.positions, axis=0).reshape(1, -1)
    np.testing.assert_array_equal(sample_particles.centroid, expected)
