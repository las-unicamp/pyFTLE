# **pyFTLE: A Python Package for Computing Finite-Time Lyapunov Exponents**

[![Python Code Quality](https://github.com/las-unicamp/pyFTLE/actions/workflows/tests.yaml/badge.svg)](https://github.com/las-unicamp/pyFTLE/actions/workflows/tests.yaml)
[![Python Code Quality](https://github.com/las-unicamp/pyFTLE/actions/workflows/code-style.yaml/badge.svg)](https://github.com/las-unicamp/pyFTLE/actions/workflows/code-style.yaml)
[![Documentation Status](https://readthedocs.org/projects/pyftle/badge/?version=latest)](https://pyftle.readthedocs.io/en/latest/)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17583865.svg)](https://doi.org/10.5281/zenodo.17583865)

`pyFTLE` computes hyperbolic Lagrangian Coherent Structures (LCS) from velocity flow field data using Finite-Time Lyapunov Exponents (FTLE).

---

## **OVERVIEW**

pyFTLE is a modular, high-performance package for computing FTLE fields. It tracks particle positions over time by integrating trajectories in a velocity field. Then, the flow map Jacobian is computed, and the largest eigenvalue of the Cauchy-Green deformation tensor determines the FTLE field.

<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/las-unicamp/pyFTLE/main/.github/ftle.gif" alt="FTLE field over airfoil" width="100%">
      <br>
      <em>Figure 1: FTLE field over an airfoil.</em>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/las-unicamp/pyFTLE/main/.github/ftle_3d_abc_flow.gif" alt="3D ABC flow FTLE field" width="100%">
      <br>
      <em>Figure 2: FTLE field of a 3D ABC flow.</em>
    </td>
  </tr>
</table>

### **Key Features**
- Supports both 2D and 3D velocity fields (structured or unstructured).
- Parallel computation of FTLE fields.
- Flexible particle integration strategies.
- Multiple velocity interpolation methods for particle positions.
- SIMD-optimized C++ backend for efficient 2D and 3D interpolations on regular grids.
- Extensible design supporting multiple file formats.
- Modular, well-structured codebase for easy customization and extension.

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
2. Install the package (this automatically installs the SIMD-optimized C++/Eigen backend):
   ```bash
   uv tool install .
   ```

After installation, `pyftle` is available globally.
This means you can import it in Python scripts or notebooks using:
```python
import pyftle
```
and also run the CLI tool directly from the terminal with:
```
pyftle -c config.yaml
```


To uninstall, just run `uv tool uninstall pyftle`

#### **Using Docker**

You can run `pyFTLE` inside a Docker container without installing dependencies locally. The Docker image includes all required dependencies and the compiled C++ extension.

**Building the Docker Image**

From the repository root directory:

```bash
docker pull lasunicamp/pyftle:latest
```
Refer to [section Running pyFTLE with Docker](#running-pyftle-with-docker) for more details about the Docker workflow.

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
from the file system (HD/SSD). In this case, the **[file-based CLI](#running-the-code-via-cli)** offers greater convenience and flexibility.

> [!NOTE]
> At present, the solver accepts MATLAB (.mat) files as input and exports results in MATLAB (.mat) or VTK (.vts, .vtp) formats.
> File I/O is implemented using SciPy, so MATLAB itself is not required.
> The modular I/O subsystem allows developers to integrate additional file formats with minimal changes.

> [!IMPORTANT]
> Input data is provided in MATLAB file format, grouped into three types: velocity, coordinate, and particle files.
>
> - **Velocity files** contain the velocity field data, where each scalar component (e.g., velocity in the x, y, and z directions) is provided in separate columns. Each column header must be properly labeled (`velocity_x`, `velocity_y`, and `velocity_z` for 3D cases), with the corresponding scalar velocity values at each point in the grid.
>
> - **Coordinate files** specify the positions where the velocity measurements were taken. The headers must correspond to the spatial coordinates (`coordinate_x`, `coordinate_y`, and `coordinate_z` for 3D cases). These coordinates map directly to the points where the velocity field data in the corresponding velocity file is measured.
>
> - **Particle files** define groups of neighboring particles used to calculate the FTLE field and more precisely compute the deformation of the Cauchy-Green tensor. In contrast to the other files, each row in the particle file contains a set of coordinates (a tuple of `[float, float]` for 2D, or `[float, float, float]` for 3D). The columns specify the relative positions of particles in the group, and the values represent the coordinates of neighboring particles. These tuples help to define the spatial relationships that are critical for computing tensor deformations in the flow field. The neighboring particles are illustrated in the accompanying figure.
>
> This structure ensures that the velocity data, coordinate information, and neighboring particle relations are clearly organized and ready for FTLE computation.

<table>
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/las-unicamp/pyFTLE/main/.github/particles.png" alt="Particles Group Image" width="450">
      <br>
      <em>Figure 3: A single group of neighboring particles.</em>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/las-unicamp/pyFTLE/main/.github/integration.gif" alt="Particle tracking over airfoil" width="450">
      <br>
      <em>Figure 4: Particles centroids being tracked.</em>
    </td>
  </tr>
</table>

Instead of passing individual MATLAB files directly to the solver, the interface expects a set of
plain text (`.txt`) files—one for each data type: velocity, coordinate, and particle data. Each of
these `.txt` files should contain a list of file paths to the corresponding `.mat` files, with one path
per line. For example, the velocity `.txt` file will list all the velocity MATLAB files (one per line),
and similarly for the coordinate and particle `.txt` files. This approach allows the solver to process
sequences of time-resolved data more easily and keeps the input interface clean and scalable.

> [!TIP]
> - The `create_list_of_input_files.py` facilitates the creation of these `.txt` files.
> - An complete example of file-based I/O workflow is provided in the Jupyter Notebooks in the `example/` folder.


### **Running the code via CLI**

The script requires several parameters, which can be passed through the command line or a configuration
file (`config.yaml`) located in the root directory. Among these parameters are `.txt` files that
indicates the location of the input files in Matlab format (the velocity field, coordinates and
particles).

> [!TIP]
> Once the parameters are properly set, the solver can be executed from the root directory with the following command:
>
> ```bash
> pyftle -c config.yaml
> ```

Alternatively, you can run the script from the CLI as:

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
    --flow_grid_shape 100,100,100 \  # comment this line for unstructured data
    --particles_grid_shape 100,100,100  # comment this line for unstructured data
```

For VSCode users, the script execution can be streamlined via `.vscode/launch.json`.


<br>

To see the complete list of command-line options and their descriptions, simply run:
```bash
pyftle --help
# or equivalently
pyftle -h
```
This will display all available parameters, their default values, and usage examples directly in the terminal.


<details>
<summary><b>⚙️ Full List of CLI Parameters (click to expand)</b></summary>


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


### **FTLE Computation Details**

The parameters `snapshot_timestep` and `flow_map_period` together define the temporal window used to compute each FTLE field.
The number of FTLE fields that will be produced is determined by the number of available velocity snapshots and the integration period of the flow map:

`N_FTLE = N_snapshots - (flow_map_period / snapshot_timestep)`

For example, suppose:

* `list_velocity_files` lists **100** velocity files,
* `snapshot_timestep = 0.01`, and
* `flow_map_period = 0.1`.

In this case, each FTLE field requires **10** consecutive velocity snapshots to integrate the flow map, so only
**90 FTLE fields** will be computed (one for each valid starting snapshot).

This ensures that the temporal integration for each FTLE remains consistent with the specified flow map period.


### **Running pyFTLE with Docker**

<details>
<summary><b>(click to expand)</b></summary>

Run the `pyftle` CLI tool directly:

```bash
docker run --rm lasunicamp/pyftle:latest --help
```

Run with a configuration file (mount your data directory):

```bash
docker run --rm \
    -v /path/to/your/data:/data \
    -w /data \
    lasunicamp/pyftle:latest -c config.yaml
```

Run with command-line arguments:

```bash
docker run --rm \
    -v /path/to/your/data:/data \
    -w /data \
    lasunicamp/pyftle:latest \
    --experiment_name "my_experiment" \
    --list_velocity_files "velocity_files.txt" \
    --list_coordinate_files "coordinate_files.txt" \
    --list_particle_files "particle_files.txt" \
    --snapshot_timestep 0.1 \
    --flow_map_period 5.0 \
    --integrator "rk4" \
    --interpolator "cubic" \
    --num_processes 4 \
    --output_format "vtk"
```

**Interactive Usage**

Start an interactive shell inside the container:

```bash
docker run -it --rm --entrypoint sh lasunicamp/pyftle:latest
```

Inside the container, you can run:
```bash
pyftle --help
python -c "import pyftle; from pyftle import AnalyticalSolver"
```

> [!TIP]
> Use volume mounts (`-v`) to access your data files and configuration files from inside the container. The `-w` flag sets the working directory inside the container.
>
> **Examples:**
>
> ```bash
> # Mount current directory and set it as working directory
> docker run --rm \
>     -v $(pwd):/data \
>     -w /data \
>     lasunicamp/pyftle:latest -c config.yaml
>
> # Mount specific data directory
> docker run --rm \
>     -v /path/to/velocity/data:/data/velocity \
>     -v /path/to/output:/data/output \
>     -w /data \
>     lasunicamp/pyftle:latest \
>     --experiment_name "my_experiment" \
>     --list_velocity_files "velocity/velocity_files.txt" \
>     --list_coordinate_files "velocity/coordinate_files.txt" \
>     --list_particle_files "velocity/particle_files.txt" \
>     --snapshot_timestep 0.1 \
>     --flow_map_period 5.0
>
> # Mount configuration file from host
> docker run --rm \
>     -v $(pwd)/config.yaml:/app/config.yaml \
>     -w /app \
>     lasunicamp/pyftle:latest -c config.yaml
> ```

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

When contributing to this repository, please make sure to follow the guidelines from the [CONTRIBUTING file](https://github.com/las-unicamp/pyFTLE/blob/main/CONTRIBUTING.md).

To make sure the Language Server Protocol (LSP) is going to work as expected, one can install the package as follows:

1. Install dependencies using UV:
   ```bash
   uv sync --all-extras
   ```
2. Install `src/` directory as an editable package:
   ```bash
   uv pip install -e '.[dev,test]' --verbose
   ```
   - This installs `src/` as an editable package, allowing you to import modules directly and modify the code during development.
   - The command also automatically installs the SIMD-optimized C++/Eigen backend.
   - Installing in editable mode helps avoid common import issues during development.


We use `pytest` for unit tests. To run the entire test suit, we recommend the following command in the base directory of the repository:
```bash
PYTHONPATH=${PWD} uv run python -m pytest
```

### **For Developers Using Docker**

If you are using Docker, you might want to build the docker image locally:

```bash
docker build -t pyftle:latest .
```

All commands shown in [section Running pyFTLE with Docker](#running-pyftle-with-docker) will work, but we need to remove the `lasunicamp/` from it to tell Docker to use the source code version and not the version from the Docker Hub.


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
