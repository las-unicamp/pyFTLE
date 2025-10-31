# **pyFTLE: A Python Package for Computing Finite-Time Lyapunov Exponents**

[![Python Code Quality](https://github.com/las-unicamp/pyFTLE/actions/workflows/tests.yaml/badge.svg)](https://github.com/las-unicamp/pyFTLE/actions/workflows/tests.yaml)
[![Python Code Quality](https://github.com/las-unicamp/pyFTLE/actions/workflows/code-style.yaml/badge.svg)](https://github.com/las-unicamp/pyFTLE/actions/workflows/code-style.yaml)

[![DOI](https://zenodo.org/badge/931585059.svg)](https://doi.org/10.5281/zenodo.17497582)

`pyFTLE` computes hyperbolic Lagrangian Coherent Structures (LCS) from velocity flow field data using Finite-Time Lyapunov Exponents (FTLE).

---

## **OVERVIEW**

pyFTLE is a modular, high-performance package for computing FTLE fields. It tracks particle positions over time by integrating trajectories in a velocity field. Then, the flow map Jacobian is computed, and the largest eigenvalue of the Cauchy-Green deformation tensor determines the FTLE field.

<div align="center">
  <table border="0" cellspacing="0" cellpadding="0">
    <tr>
      <td style="text-align: center; width: 45%;">
        <img src="https://github.com/las-unicamp/pyFTLE/blob/main/.github/ftle.gif" alt="FTLE field over airfoil" width="100%">
      </td>
      <td style="text-align: center; width: 45%;">
        <img src="https://github.com/las-unicamp/pyFTLE/blob/main/.github/ftle_3d_abc_flow.gif" alt="3D ABC flow FTLE field" width="100%">
      </td>
    </tr>
    <tr>
      <td style="text-align: center; width: 45%;">
        <em>Figure 1: FTLE field over an airfoil.</em>
      </td>
      <td style="text-align: center; width: 45%;">
        <em>Figure 2: FTLE field of a 3D ABC flow.</em>
      </td>
    </tr>
  </table>
</div>


### **Key Features**
- Supports both 2D and 3D velocity fields (structured or unstructured).
- Parallel computation of FTLE fields.
- Flexible particle integration strategies.
- Multiple velocity interpolation methods for particle positions.
- SIMD-optimized C++ backend for efficient 2D and 3D interpolations on regular grids.
- Extensible design supporting multiple file formats.
- Modular, well-structured codebase for easy customization and extension.

> [!NOTE]
> The current implementation supports MATLAB files for input and MATLAB or VTK files for output. Thanks to its modular architecture, additional file formats can be integrated with minimal effort.

---

## **INSTALLATION**

### **Requirements**
- Python 3.10+

### **Using UV (Recommended)**

[UV](https://docs.astral.sh/uv/) is a modern Python package and project manager that simplifies dependency management.

#### **Installation Steps:**
1. Clone the repository:
   ```bash
   git clone https://github.com/las-unicamp/pyFTLE.git
   cd pyFTLE
   ```
2. Install dependencies using UV:
   ```bash
   uv sync --all-extras
   ```
3. Install `src/` directory as an editable package:
   ```bash
   uv pip install -e '.[dev,test]' --verbose
   ```
   - This installs `src/` as an editable package, allowing you to import modules directly and modify the code during development.
   - The command also automatically installs the SIMD-optimized C++/Eigen backend.
   - Installing in editable mode helps avoid common import issues during development.

---

## **USAGE**

The code features both a clean, CLI-oriented architecture (utilizing configuration files and
file-based I/O) and a lightweight, notebook-friendly API. The latter allows you to run small-scale
examples entirely in memory, eliminating the need to handle intermediate files, which makes it
perfect for demonstrating in Jupyter notebooks. Several such notebooks, located in the `examples/`
folder, combine analytical velocity fields with visual explanations to illustrate the FTLE
solver’s execution.

> [!TIP]
> For production runs, it is often more practical to read velocity and coordinate data directly
from the file system (HD/SSD). In this case, the [file-based CLI](#anchor-point-running-via-CLI) offers greater convenience and flexibility.


> [!IMPORTANT]
> As previously mentioned, the current implementation only supports MATLAB file formats for input data. These files consist of three types: velocity, coordinate, and particle files.
>
> - **Velocity files** contain the velocity field data, where each scalar component (e.g., velocity in the x, y, and z directions) is provided in separate columns. Each column header must be properly labeled (`velocity_x`, `velocity_y`, and `velocity_z` for 3D cases), with the corresponding scalar velocity values at each point in the grid.
>
> - **Coordinate files** specify the positions where the velocity measurements were taken. The headers must correspond to the spatial coordinates (`coordinate_x`, `coordinate_y`, and `coordinate_z` for 3D cases). These coordinates map directly to the points where the velocity field data in the corresponding velocity file is measured.
>
> - **Particle files** define groups of neighboring particles used to calculate the FTLE field and more precisely compute the deformation of the Cauchy-Green tensor. In contrast to the other files, each row in the particle file contains a set of coordinates (a tuple of `[float, float]` for 2D, or `[float, float, float]` for 3D). The columns specify the relative positions of particles in the group, and the values represent the coordinates of neighboring particles. These tuples help to define the spatial relationships that are critical for computing tensor deformations in the flow field. The neighboring particles are illustrated in the accompanying figure.
>
> This structure ensures that the velocity data, coordinate information, and neighboring particle relations are clearly organized and ready for FTLE computation.


<div align="center">
  <table border="0" cellspacing="0" cellpadding="0">
    <tr>
      <td style="text-align: center; width: 45%;">
        <img src="https://github.com/las-unicamp/pyFTLE/blob/main/.github/particles.png" alt="Particles Group Image" width="100%">
      </td>
      <td style="text-align: center; width: 45%;">
        <img src="https://github.com/las-unicamp/pyFTLE/blob/main/.github/integration.gif" alt="Particle tracking over airfoil" width="100%">
      </td>
    </tr>
    <tr>
      <td style="text-align: center; width: 45%;">
        <em>Figure 3: A single group of neighboring particles.</em>
      </td>
      <td style="text-align: center; width: 45%;">
        <em>Figure 4: Particles centroids being tracked.</em>
      </td>
    </tr>
  </table>
</div>



Instead of passing individual MATLAB files directly to the solver, the interface expects a set of
plain text (`.txt`) files—one for each data type: velocity, coordinate, and particle data. Each of
these `.txt` files should contain a list of file paths to the corresponding `.mat` files, with one path
per line. For example, the velocity `.txt` file will list all the velocity MATLAB files (one per line),
and similarly for the coordinate and particle `.txt` files. This approach allows the solver to process
sequences of time-resolved data more easily and keeps the input interface clean and scalable.

> [!TIP]
> - The `create_list_of_input_files.py` facilitates the creation of these `.txt` files.
> - An complete example of file-based I/O workflow is provided in the Jupyter Notebooks in the `example/` folder.


<a name="anchor-point-running-via-CLI"></a>

### **Running the code via CLI**

The script requires several parameters, which can be passed through the command line or a configuration
file (`config.yaml`) located in the root directory. Among these parameters are `.txt` files that
indicates the location of the input files in Matlab format (the velocity field, coordinates and
particles).

> [!TIP]
> Once the parameters are properly set, the solver can be executed from the root directory with the following command:
>
> ```bash
> PYTHONPATH=${PWD} uv run python src/app.py -c config.yaml
> ```

Alternatively, you can run the script from the CLI as:

```bash
PYTHONPATH=${PWD} uv run python app.py \
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
    --flow_grid_shape 100,100,100 \  # comment this line for unstructured data
    --particles_grid_shape 100,100,100  # comment this line for unstructured data
```

For VSCode users, the script execution can be streamlined via `.vscode/launch.json`.


<details>
<summary><b>⚙️ Full List of CLI Parameters (click to expand)</b></summary>

<br>

### **Required Parameters**

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

### **Optional Parameters**

| Parameter               | Type        | Description                                                                                     |
| ----------------------- | ----------- | ----------------------------------------------------------------------------------------------- |
| `flow_grid_shape`       | `list[int]` | Grid shape for structured velocity measurements. It must be a comma-separated list of integers. |
| `particles_grid_shape`  | `list[int]` | Grid shape for structured particle points. It must be a comma-separated list of integers.       |


Interpolation behavior depends on whether your velocity data is structured or unstructured:

- If `flow_grid_shape` is **not provided**, the velocity field is treated as **unstructured**.
  In this case, you can use the `cubic`, `linear`, or `nearest` interpolators, which rely on Delaunay triangulations.
  This approach offers flexibility but comes with higher computational cost.

- If `flow_grid_shape` **is provided**, the velocity field is considered **structured**.
  You can still choose `cubic`, `linear`, or `nearest`, but interpolation becomes significantly faster because it exploits the rectilinear grid structure of the data.

- For **maximum performance** on regular structured grids, `pyFTLE` includes custom **bi- and trilinear interpolators** implemented in **C++/Eigen**, achieving up to **10× speedup** compared to SciPy’s implementation.
  To use this optimized backend, specify `flow_grid_shape` and set `interpolator` to `grid`.

The parameter `particles_grid_shape` is optional and mainly affects how results are written to disk.
If the particle centroids form a regular grid, defining this parameter enables structured output—making post-processing and visualization more straightforward.

</details>

---


## **REFERENCES**

A list of scientific works using pyFTLE includes:

1. [de Souza, Miotto, Wolf. _Active flow control of vertical-axis wind turbines: Insights from large-eddy simulation and finite-time resolvent analysis_. Journal of Fluids and Structures, 2025.](https://doi.org/10.1016/j.jfluidstructs.2025.104410)
2. [de Souza, Wolf, Safari, Yeh. _Control of Deep Dynamic Stall by Duty-Cycle Actuation Informed by Stability Analysis_. AIAA Journal, 2025.](https://doi.org/10.2514/1.J064980)
3. Lui, Wolf. _Interplay between streaks and vortices in shock-boundary layer interactions with conditional bubble events over a turbine airfoil_. Physical Review Fluids, 2025.
4. [Lui, Wolf, Ricciardi, Gaitonde. _Analysis of Streamwise Vortices in a Supersonic Turbine Cascade_. AIAA Aviation Forum and Ascend, 2024.](https://doi.org/10.2514/6.2024-3800)


---

## **LICENSE**

This project is licensed under the **MIT License**.

---

## **CONTRIBUTING**

When contributing to this repository, please make sure to follow the guidelines from the [CONTRIBUTING file](CONTRIBUTING.md).

We use `pytest` for unit tests. To run the entire test suit, we recommend the following command in the base directory of the repository:
```bash
PYTHONPATH=${PWD} uv run python -m pytest
```

---

## **FUNDING**

The authors acknowledge Fundação de Amparo à Pesquisa do Estado de São Paulo, FAPESP, for supporting the present work under research grants No. 2013/08293-7, 2019/17874-0, 2021/06448-0, 2022/09196-4, 2022/08567-9, and 2024/20547-9. Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq) is also acknowledged for supporting this research under grant No. 304320/2024-2.

---

## **MAIN DEVELOPERS**

- **Renato Fuzaro Miotto**
- **Lucas Feitosa de Souza**
- **William Roberto Wolf**

---

For bug reports, feature requests, or contributions, please open an issue or submit a pull request on [GitHub](https://github.com/las-unicamp/pyFTLE).
