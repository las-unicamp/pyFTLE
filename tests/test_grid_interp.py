import numpy as np

from pyftle import ginterp


# ============================================================
# Helper functions to generate known grids
# ============================================================
def generate_2d_grid(nx=5, ny=5):
    """Generate a simple 2D grid v(x, y) = x + 2*y."""
    x = np.arange(nx)
    y = np.arange(ny)
    xv, yv = np.meshgrid(x, y, indexing="ij")
    v = xv + 2 * yv
    return v.astype(float)


def generate_3d_grid(nx=5, ny=5, nz=5):
    """Generate a simple 3D grid v(x, y, z) = x + 2*y + 3*z."""
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    v = xv + 2 * yv + 3 * zv
    return v.astype(float)


# ============================================================
# 2D Tests
# ============================================================
def test_interp2d_vec_exact_values():
    v = generate_2d_grid()
    points = np.array([[1.5, 2.5], [0.2, 0.8]], dtype=float)

    # Analytical function: f(x, y) = x + 2*y
    expected = points[:, 0] + 2 * points[:, 1]

    out = ginterp.interp2d_vec(v, points)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_interp2d_vec_inplace_matches_vec():
    v = generate_2d_grid()
    points = np.random.rand(10, 2) * 3.0
    out_vec = ginterp.interp2d_vec(v, points)

    out_inplace = np.empty(points.shape[0])
    ginterp.interp2d_vec_inplace(v, points, out_inplace)

    np.testing.assert_allclose(out_vec, out_inplace, rtol=1e-12, atol=1e-12)


def test_interp2d_out_of_bounds_returns_zero():
    v = generate_2d_grid()
    points = np.array([[10.0, 10.0]])  # Outside bounds
    out = ginterp.interp2d_vec(v, points)
    assert np.all(out == 0.0)


# ============================================================
# 3D Tests
# ============================================================
def test_interp3d_vec_exact_values():
    v = generate_3d_grid()
    points = np.array([[1.5, 2.5, 3.5], [0.2, 0.8, 0.1]], dtype=float)

    # Analytical function: f(x, y, z) = x + 2*y + 3*z
    expected = points[:, 0] + 2 * points[:, 1] + 3 * points[:, 2]

    out = ginterp.interp3d_vec(v, points)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_interp3d_vec_inplace_matches_vec():
    v = generate_3d_grid()
    points = np.random.rand(10, 3) * 3.0
    out_vec = ginterp.interp3d_vec(v, points)

    out_inplace = np.empty(points.shape[0])
    ginterp.interp3d_vec_inplace(v, points, out_inplace)

    np.testing.assert_allclose(out_vec, out_inplace, rtol=1e-12, atol=1e-12)


def test_interp3d_out_of_bounds_returns_zero():
    v = generate_3d_grid()
    points = np.array([[10.0, 10.0, 10.0]])  # Outside bounds
    out = ginterp.interp3d_vec(v, points)
    assert np.all(out == 0.0)


# ============================================================
# General Shape & Type Checks
# ============================================================
def test_output_shape_and_type_2d():
    v = generate_2d_grid()
    points = np.random.rand(20, 2) * 3.0
    out = ginterp.interp2d_vec(v, points)
    assert out.shape == (20,)
    assert out.dtype == np.float64


def test_output_shape_and_type_3d():
    v = generate_3d_grid()
    points = np.random.rand(30, 3) * 3.0
    out = ginterp.interp3d_vec(v, points)
    assert out.shape == (30,)
    assert out.dtype == np.float64
