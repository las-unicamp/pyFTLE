// GridInterp_fast.cpp
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <stdexcept>

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace py = pybind11;

void interp3d_vec_inplace(py::array_t<double, py::array::c_style | py::array::forcecast> v,
                          py::array_t<double, py::array::c_style | py::array::forcecast> points,
                          py::array_t<double, py::array::c_style | py::array::forcecast> out) {
  // Ensure contiguous and correct dims
  if (v.ndim() != 3) throw std::runtime_error("v must be 3D");
  if (points.ndim() != 2 || points.shape(1) != 3)
    throw std::runtime_error("points must have shape (N,3)");
  if (out.ndim() != 1) throw std::runtime_error("out must be 1D");

  const ssize_t N = points.shape(0);
  if (out.shape(0) != N) throw std::runtime_error("out length must equal points.shape[0]");

  // Shapes
  const int X = static_cast<int>(v.shape(0));
  const int Y = static_cast<int>(v.shape(1));
  const int Z = static_cast<int>(v.shape(2));

  // Pointers to raw data
  const double *v_ptr = static_cast<const double *>(v.request().ptr);
  const double *pts_ptr = static_cast<const double *>(points.request().ptr);
  double *out_ptr = static_cast<double *>(out.request().ptr);

  const int YZ = Y * Z;

  for (ssize_t i = 0; i < N; ++i) {
    const double px = pts_ptr[i * 3 + 0];
    const double py = pts_ptr[i * 3 + 1];
    const double pz = pts_ptr[i * 3 + 2];

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

    // Matches the Cython condition: x0 in [0, X-2], etc.
    if (x0 >= 0 && x1 < X && y0 >= 0 && y1 < Y && z0 >= 0 && z1 < Z) {
      auto idx = [&](int xi, int yi, int zi) -> ssize_t {
        return static_cast<ssize_t>(xi) * YZ + static_cast<ssize_t>(yi) * Z +
               static_cast<ssize_t>(zi);
      };

      const double v_x0y0z0 = v_ptr[idx(x0, y0, z0)];
      const double v_x1y0z0 = v_ptr[idx(x1, y0, z0)];
      const double v_x0y0z1 = v_ptr[idx(x0, y0, z1)];
      const double v_x1y0z1 = v_ptr[idx(x1, y0, z1)];
      const double v_x0y1z0 = v_ptr[idx(x0, y1, z0)];
      const double v_x1y1z0 = v_ptr[idx(x1, y1, z0)];
      const double v_x0y1z1 = v_ptr[idx(x0, y1, z1)];
      const double v_x1y1z1 = v_ptr[idx(x1, y1, z1)];

      const double c00 =
          static_cast<double>(v_x0y0z0) * (1.0 - xd) + static_cast<double>(v_x1y0z0) * xd;
      const double c01 =
          static_cast<double>(v_x0y0z1) * (1.0 - xd) + static_cast<double>(v_x1y0z1) * xd;
      const double c10 =
          static_cast<double>(v_x0y1z0) * (1.0 - xd) + static_cast<double>(v_x1y1z0) * xd;
      const double c11 =
          static_cast<double>(v_x0y1z1) * (1.0 - xd) + static_cast<double>(v_x1y1z1) * xd;

      const double c0 = c00 * (1.0 - yd) + c10 * yd;
      const double c1 = c01 * (1.0 - yd) + c11 * yd;
      c = c0 * (1.0 - zd) + c1 * zd;
    } else {
      c = 0.0;
    }

    out_ptr[i] = c;
  }
}

void interp2d_vec_inplace(py::array_t<double, py::array::c_style | py::array::forcecast> v,
                          py::array_t<double, py::array::c_style | py::array::forcecast> points,
                          py::array_t<double, py::array::c_style | py::array::forcecast> out) {
  if (v.ndim() != 2) throw std::runtime_error("v must be 2D");
  if (points.ndim() != 2 || points.shape(1) != 2)
    throw std::runtime_error("points must have shape (N,2)");
  if (out.ndim() != 1) throw std::runtime_error("out must be 1D");

  const ssize_t N = points.shape(0);
  if (out.shape(0) != N) throw std::runtime_error("out length must equal points.shape[0]");

  // --- Grid dimensions ---
  const int nx = static_cast<int>(v.shape(0));  // x-dimension
  const int ny = static_cast<int>(v.shape(1));  // y-dimension

  const double *v_ptr = static_cast<const double *>(v.request().ptr);
  const double *pts_ptr = static_cast<const double *>(points.request().ptr);
  double *out_ptr = static_cast<double *>(out.request().ptr);

  for (ssize_t i = 0; i < N; ++i) {
    const double px = pts_ptr[i * 2 + 0];  // x coordinate
    const double py = pts_ptr[i * 2 + 1];  // y coordinate

    const int x0 = static_cast<int>(std::floor(px));
    const int y0 = static_cast<int>(std::floor(py));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const double xd = px - x0;
    const double yd = py - y0;

    double c = 0.0;

    if (x0 >= 0 && x1 < nx && y0 >= 0 && y1 < ny) {
      auto idx = [&](int xi, int yi) -> ssize_t {
        return static_cast<ssize_t>(xi) * ny + static_cast<ssize_t>(yi);
      };

      const double v00 = v_ptr[idx(x0, y0)];
      const double v10 = v_ptr[idx(x1, y0)];
      const double v01 = v_ptr[idx(x0, y1)];
      const double v11 = v_ptr[idx(x1, y1)];

      c = v00 * (1.0 - xd) * (1.0 - yd) + v10 * xd * (1.0 - yd) + v01 * (1.0 - xd) * yd +
          v11 * xd * yd;
    }

    out_ptr[i] = c;
  }
}

PYBIND11_MODULE(ginterp, m) {
  m.doc() = "Bi and trilinear interpolation (float64 grid, in-place output)";
  m.def("interp3d_vec_inplace", &interp3d_vec_inplace, py::arg("v"), py::arg("points"),
        py::arg("out"));
  m.def("interp2d_vec_inplace", &interp2d_vec_inplace, py::arg("v"), py::arg("points"),
        py::arg("out"));
}
