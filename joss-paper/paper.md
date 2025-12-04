---
title: 'pyFTLE: A Python Package for Computing Finite-Time Lyapunov Exponents'
tags:
  - Python
  - fluid dynamics
  - finite-time lyapunov exponents
  - lagrangian coherent structures
authors:
  - name: Renato Fuzaro Miotto
    orcid: 0000-0002-7813-8873
    equal-contrib: true
    affiliation: 1
    corresponding: true
  - name: Lucas Feitosa de Souza
    orcid: 0000-0002-1394-4479
    equal-contrib: true
    affiliation: 1
  - name: William Roberto Wolf
    orcid: 0000-0001-8207-8466
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: School of Mechanical Engineering, University of Campinas, Brazil
   index: 1
   ror: 04wffgt70
date: 04 December 2025
bibliography: paper.bib

---

# Summary

The Finite-Time Lyapunov Exponent (FTLE) field is a fundamental quantity for analysis
of dynamical systems, with particular applications in fluid mechanics and transport
phenomena. It enables the identification of Lagrangian Coherent Structures
(LCSs) [@Haller_annual_review] by characterizing the trajectories of attracting, repelling
and shearing material surfaces in manifolds. Hence, the FTLE plays a crucial role in
elucidating transport mechanisms and identifying separatrices in fluid flows [@Brunton_LCS].
This work presents a Python-based solver for computing FTLE fields from velocity data, featuring
optimized integration techniques and parallel computations.

The proposed implementation is designed for high performance using Numba, along with a
SIMD-optimized C++ backend for efficient 2D and 3D interpolations on regular grids.
These optimizations allow for fast Lagrangian particle advection and efficient gradient
estimation. The solver is flexible and modular, supporting structured and unstructured grids
and arbitrary velocity fields while maintaining a well-documented and user-friendly interface.

# Statement of Need

Understanding transport barriers and coherent structures in unsteady flows is crucial in various
scientific and engineering applications such as oceanography, meteorology, and aerodynamics.
Despite the importance of FTLE computations, existing implementations are often specialized,
computationally expensive, or lack extensibility.

`pyFTLE` addresses these limitations by providing a modular, high-performance package that:

- Supports both 2D and 3D velocity fields (structured or unstructured);
- Enables parallelized FTLE computations for improved performance;
- Offers flexible particle integration strategies (RK4, Euler, AB2);
- Provides multiple velocity interpolation methods for particle positions;
- Includes a SIMD-optimized C++ backend for efficient 2D and 3D interpolations on regular grids;
- Features an extensible design supporting multiple file formats;
- Offers a modular, well-structured codebase for easy customization and extension.

# Implementation

Initially, a grid of particles $X_{0} \subset \mathbb{R}^{2}$ is established across the
domain of interest. These particles are integrated in the velocity field from the initial
time $0$ to the final time $T$, yielding a time-T particle flow map denoted as $\Phi_{0}^{T}$
defined as follows:

$$
\Phi_{0}^{T}: \mathbb{R}^{2} \to \mathbb{R}^{2}; \mathbf{x}(0) \mapsto \mathbf{x}(0) + \int_{o}^{T} \mathbf{u}(\mathbf{x}(\tau), \tau)  d\tau \mbox{ .}
$$

Here, $\mathbf{u}(\mathbf{x}(\tau), \tau)$ denotes the time-dependent velocity field over the
particle trajectory $\mathbf{x}(\tau)$ at a time $\tau$. The flow map Jacobian $\mathbf{D} \Phi_{0}^{T}$
is then computed by a central finite difference scheme using the neighbouring particles in a Cartesian
mesh such as:

$$
\mathbf{D} \Phi_{0}^{T} = \left[
\begin{array}{cc}
\frac{\Delta x(T)}{ \Delta x (0)} & \frac{\Delta x(T)}{ \Delta y (0)} \\
\frac{\Delta y(T)}{ \Delta x (0)} & \frac{\Delta y(T)}{ \Delta y (0)}
\end{array}
\right] = \left[
\begin{array}{cc}
\frac{x_{i+1,j}(T)- x_{i-1,j} (T) }{ x_{i+1,j}(0)- x_{i-1,j}(0) } &  \frac{x_{i,j+1}(T)- x_{i,j-1} (T) }{ y_{i,j+1}(0)- y_{i,j-1}(0)}\\
\frac{y_{i+1,j}(T)- y_{i-1,j} (T) }{ x_{i+1,j}(0)- x_{i-1,j}(0) } &  \frac{y_{i,j+1}(T)- y_{i,j-1} (T) }{ y_{i,j+1}(0)- y_{i,j-1}(0) }
\end{array}
\right] \mbox{ ,}
$$

where $x$ and $y$ denote the particle coordinates and subscripts $i$ and $j$ denote their indices
in the computational domain. Finally, the Cauchy-Green deformation tensor is computed as:

$$
\boldsymbol{\Delta} = (\mathbf{D} \Phi_{0}^{T})^{\ast} \mathbf{D} \Phi_{0}^{T} \mbox{ .}
$$

Here, $\ast$ denotes the transpose and the largest eigenvalue $(\lambda_{max})$ from this tensor is computed to form the FTLE field,

$$
\sigma (\mathbf{D} \Phi_{0}^{T} ; \mathbf{x_{0}}) = \frac{1}{\lvert T \rvert} \log{ \sqrt{\lambda_{max}(\boldsymbol{\Delta} (\mathbf{x_{0}}))}} \mbox{ .}
$$

The FTLEs are computed using an auxiliary grid in which the flow properties are interpolated on.
\autoref{fig:numerical_method} presents an example of the spatial discretization approach used in the
present study. A reference mesh (indicated by the red dots in \autoref{fig:numerical_method}) is first
placed on top of the flow grid and the overlapping points on the airfoil solid surface are removed.
The auxiliary grid (represented by the black dots) is then constructed with the maximum distance allowed
from the reference points to the airfoil surface ensuring that all points remain outside the solid body.

To facilitate efficient I/O for time-resolved data, the implementation expects a set of plain text (`.txt`)
files listing paths to the data: one list for velocity files, one for coordinate files, and one for particle
files. The **velocity files** contain scalar components (e.g., `velocity_x`, `velocity_y`), while
**coordinate files** specify the measurement locations (`coordinate_x`, `coordinate_y`).

Crucially, the **particle files** define groups of neighboring particles used to calculate the flow map
Jacobian. Each row in the particle file contains a set of coordinates (tuples of `[float, float]` for 2D)
defining the `left`, `right`, `top`, and `bottom` (and `front` and `bach` for 3D) neighbors surrounding a
central location. This structure ensures that spatial relationships required for tensor deformation are
clearly organized.

![Auxiliary grid used to compute the flow map Jacobian. The properties at the reference point marked in red are computed by integrating then performing the central finite difference of the auxiliary grid points in black.\label{fig:numerical_method}](fig1.png){ width=70% }

The FTLE fields can be computed by integrating the particles in forward or backward time. This choice
yields different interpretations of the LCSs providing analogs for stable (forward time integration)
and unstable manifolds (backward time integration) from dynamical systems [@Brunton_LCS; @Haller_annual_review].
In the present work we employ the backward time integration of the particles as it enables a direct measure
of material transport in forward time, mimicking experimental flow visualization by tracers.

![Comparison of the vorticity field, the FTLE field, and the FTLE field shaded by vorticity sign for a moving airfoil.\label{fig:example}](fig2.png)

# Example Usage

The solver can be executed via the command-line interface (CLI) using the `pyftle` command. It requires several
parameters, which can be passed as arguments or through a configuration file.

The following command illustrates a typical execution:

```bash
pyftle \
    --experiment_name "my_experiment" \
    --list_velocity_files "velocity_files.txt" \
    --list_coordinate_files "coordinate_files.txt" \
    --list_particle_files "particle_files.txt" \
    --snapshot_timestep 0.1 \
    --flow_map_period 5.0 \
    --integrator "rk4" \
    --interpolator "cubic" \
    --num_processes 4 \
    --output_format "vtk" \
    --flow_grid_shape 100,100,100 \
    --particles_grid_shape 100,100,100
```

Alternatively, a configuration file can be used:

```bash
python main.py -c config.yaml
```

### Parameters

| Parameter               | Type    | Description                                                                                   |
| ----------------------- | ------- | --------------------------------------------------------------------------------------------- |
| `experiment_name`       | `str`   | Name of the subdirectory where the FTLE fields will be saved.                                 |
| `list_velocity_files`   | `str`   | Path to a text file listing velocity data files.                                              |
| `list_coordinate_files` | `str`   | Path to a text file listing coordinate files.                                                 |
| `list_particle_files`   | `str`   | Path to a text file listing particle data files.                                              |
| `snapshot_timestep`     | `float` | Timestep between snapshots (positive for forward-time FTLE, negative for backward-time FTLE). |
| `flow_map_period`       | `float` | Integration period for computing the flow map.                                                |
| `integrator`            | `str`   | Time-stepping method (`euler`, `ab2`, `rk4`).                                                 |
| `interpolator`          | `str`   | Interpolation method (`cubic`, `linear`, `nearest`, `grid`).                                  |
| `num_processes`         | `int`   | Number of workers in the multiprocessing pool. Each worker computes the FTLE of a snapshot.   |
| `output_format`         | `str`   | Output format (`mat`, `vtk`).                                                                 |
| `flow_grid_shape`       | `list[int]` | (Optional) Grid shape for structured velocity measurements. It must be a comma-separated list of integers. |
| `particles_grid_shape`  | `list[int]` | (Optional) Grid shape for structured particle points. It must be a comma-separated list of integers.       |


### Optimized Interpolation

The solver's performance relies heavily on the definition of the grid shape. If `flow_grid_shape` is
provided, the velocity field is treated as structured. This enables `pyFTLE` to utilize custom
bi- and trilinear interpolators implemented in a SIMD-optimized C++/Eigen backend (by setting
interpolator to `grid`), achieving up to 10x speedup compared to standard implementations.
If `flow_grid_shape` is omitted, the data is treated as unstructured, and the solver defaults to
Delaunay triangulation-based interpolation (cubic, linear, or nearest).


# Recent works using pyFTLE

To date, the works that have utilized pyFTLE include: @Lucas_aiaaj_2025, @Lucas_jfs_2025, @SLui_2024, and @Lucas_2024


# Acknowledgements

We acknowledge Fundação de Amparo à Pesquisa do Estado de São Paulo (FAPESP) for supporting this work
under Grants No. 2013/08293-7, 2013/07375-0, 2019/17874-0, 2021/06448-0, 2022/08567-9, 2022/09196-4,
and 2024/20547-9. We also acknowledge the support of Conselho Nacional de Desenvolvimento Científico
e Tecnológico (CNPq) under grant No. 304320/2024-2.

# References
