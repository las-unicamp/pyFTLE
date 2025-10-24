#include <pybind11/eigen.h>  // <-- Enables seamless Eigen <-> NumPy conversion
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <cmath>

namespace py = pybind11;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// ============================================================
// 3D Trilinear Interpolation
// ============================================================

// -----------------------------
// Trilinear interpolation (return new array)
// -----------------------------
py::array_t<double> interp3d_vec(const py::array_t<double>& v,
                                 const py::EigenDRef<const MatrixXd>& points) {
  auto buf = v.unchecked<3>();  // Fast access to the NumPy 3D array
  const int X = buf.shape(0);
  const int Y = buf.shape(1);
  const int Z = buf.shape(2);
  const int N = points.rows();

  auto out = py::array_t<double>(N);
  auto out_mut = out.mutable_unchecked<1>();

  for (int i = 0; i < N; ++i) {
    const double px = points(i, 0);
    const double py = points(i, 1);
    const double pz = points(i, 2);

    const int x0 = static_cast<int>(std::floor(px));
    const int y0 = static_cast<int>(std::floor(py));
    const int z0 = static_cast<int>(std::floor(pz));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;

    const double xd = px - x0;
    const double yd = py - y0;
    const double zd = pz - z0;

    double c = 0.0;
    if (x0 >= 0 && x1 < X && y0 >= 0 && y1 < Y && z0 >= 0 && z1 < Z) {
      const double c00 = buf(x0, y0, z0) * (1 - xd) + buf(x1, y0, z0) * xd;
      const double c01 = buf(x0, y0, z1) * (1 - xd) + buf(x1, y0, z1) * xd;
      const double c10 = buf(x0, y1, z0) * (1 - xd) + buf(x1, y1, z0) * xd;
      const double c11 = buf(x0, y1, z1) * (1 - xd) + buf(x1, y1, z1) * xd;

      const double c0 = c00 * (1 - yd) + c10 * yd;
      const double c1 = c01 * (1 - yd) + c11 * yd;

      c = c0 * (1 - zd) + c1 * zd;
    }

    out_mut(i) = c;
  }

  return out;
}

// -----------------------------
// Trilinear interpolation (in-place version)
// -----------------------------
void interp3d_vec_inplace(const py::array_t<double>& v, const py::EigenDRef<const MatrixXd>& points,
                          py::EigenDRef<VectorXd> out) {
  auto buf = v.unchecked<3>();
  const int X = buf.shape(0);
  const int Y = buf.shape(1);
  const int Z = buf.shape(2);
  const int N = points.rows();

  if (out.size() != N) throw std::runtime_error("Output array has incorrect shape.");

  for (int i = 0; i < N; ++i) {
    const double px = points(i, 0);
    const double py = points(i, 1);
    const double pz = points(i, 2);

    const int x0 = static_cast<int>(std::floor(px));
    const int y0 = static_cast<int>(std::floor(py));
    const int z0 = static_cast<int>(std::floor(pz));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const int z1 = z0 + 1;

    const double xd = px - x0;
    const double yd = py - y0;
    const double zd = pz - z0;

    double c = 0.0;
    if (x0 >= 0 && x1 < X && y0 >= 0 && y1 < Y && z0 >= 0 && z1 < Z) {
      const double c00 = buf(x0, y0, z0) * (1 - xd) + buf(x1, y0, z0) * xd;
      const double c01 = buf(x0, y0, z1) * (1 - xd) + buf(x1, y0, z1) * xd;
      const double c10 = buf(x0, y1, z0) * (1 - xd) + buf(x1, y1, z0) * xd;
      const double c11 = buf(x0, y1, z1) * (1 - xd) + buf(x1, y1, z1) * xd;

      const double c0 = c00 * (1 - yd) + c10 * yd;
      const double c1 = c01 * (1 - yd) + c11 * yd;

      c = c0 * (1 - zd) + c1 * zd;
    }

    out(i) = c;
  }
}

// ============================================================
// 2D Bilinear Interpolation
// ============================================================

// -----------------------------
// Bilinear interpolation (return new array)
// -----------------------------
py::array_t<double> interp2d_vec(const py::array_t<double>& v,
                                 const py::EigenDRef<const MatrixXd>& points) {
  auto buf = v.unchecked<2>();
  const int X = buf.shape(0);
  const int Y = buf.shape(1);
  const int N = points.rows();

  auto out = py::array_t<double>(N);
  auto out_mut = out.mutable_unchecked<1>();

  for (int i = 0; i < N; ++i) {
    const double px = points(i, 0);
    const double py = points(i, 1);

    const int x0 = static_cast<int>(std::floor(px));
    const int y0 = static_cast<int>(std::floor(py));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const double xd = px - x0;
    const double yd = py - y0;

    double c = 0.0;
    if (x0 >= 0 && x1 < X && y0 >= 0 && y1 < Y) {
      const double c00 = buf(x0, y0);
      const double c10 = buf(x1, y0);
      const double c01 = buf(x0, y1);
      const double c11 = buf(x1, y1);

      const double c0 = c00 * (1 - xd) + c10 * xd;
      const double c1 = c01 * (1 - xd) + c11 * xd;

      c = c0 * (1 - yd) + c1 * yd;
    }

    out_mut(i) = c;
  }

  return out;
}

// -----------------------------
// Bilinear interpolation (in-place version)
// -----------------------------
void interp2d_vec_inplace(const py::array_t<double>& v, const py::EigenDRef<const MatrixXd>& points,
                          py::EigenDRef<VectorXd> out) {
  auto buf = v.unchecked<2>();
  const int X = buf.shape(0);
  const int Y = buf.shape(1);
  const int N = points.rows();

  if (out.size() != N) throw std::runtime_error("Output array has incorrect shape.");

  for (int i = 0; i < N; ++i) {
    const double px = points(i, 0);
    const double py = points(i, 1);

    const int x0 = static_cast<int>(std::floor(px));
    const int y0 = static_cast<int>(std::floor(py));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const double xd = px - x0;
    const double yd = py - y0;

    double c = 0.0;
    if (x0 >= 0 && x1 < X && y0 >= 0 && y1 < Y) {
      const double c00 = buf(x0, y0);
      const double c10 = buf(x1, y0);
      const double c01 = buf(x0, y1);
      const double c11 = buf(x1, y1);

      const double c0 = c00 * (1 - xd) + c10 * xd;
      const double c1 = c01 * (1 - xd) + c11 * xd;

      c = c0 * (1 - yd) + c1 * yd;
    }

    out(i) = c;
  }
}

// ============================================================
// pybind11 Module Definition
// ============================================================
PYBIND11_MODULE(ginterp, m) {
  m.doc() = "2D and 3D interpolation (bilinear and trilinear) using Eigen and pybind11";

  // 3D
  m.def("interp3d_vec", &interp3d_vec, "Vectorized 3D interpolation");
  m.def("interp3d_vec_inplace", &interp3d_vec_inplace, "In-place 3D interpolation");

  // 2D
  m.def("interp2d_vec", &interp2d_vec, "Vectorized 2D interpolation");
  m.def("interp2d_vec_inplace", &interp2d_vec_inplace, "In-place 2D interpolation");
}
