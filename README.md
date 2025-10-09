# **pyFTLE: A Python Package for Computing Finite-Time Lyapunov Exponents**

[![Python Code Quality](https://github.com/las-unicamp/pyFTLE/actions/workflows/tests.yaml/badge.svg)](https://github.com/las-unicamp/pyFTLE/actions/workflows/tests.yaml)
[![Python Code Quality](https://github.com/las-unicamp/pyFTLE/actions/workflows/code-style.yaml/badge.svg)](https://github.com/las-unicamp/pyFTLE/actions/workflows/code-style.yaml)

`pyFTLE` computes hyperbolic Lagrangian Coherent Structures (LCS) from velocity flow field data using Finite-Time Lyapunov Exponents (FTLE).

---

## **OVERVIEW**

<div align="center">
  <img src="https://github.com/las-unicamp/pyFTLE/blob/main/.github/integration.gif" width="45%" />
  <img src="https://github.com/las-unicamp/pyFTLE/blob/main/.github/ftle.gif" width="45%" />
</div>
<div align="center">
  <img src="https://github.com/las-unicamp/pyFTLE/blob/main/.github/ftle_3d_abc_flow.gif" width="45%" />
</div>


pyFTLE provides a robust and modular implementation for computing FTLE fields. It tracks particle positions over time by interpolating a given velocity field and integrating their motion. After a specified integration period, the flow map Jacobian is computed, and the largest eigenvalue of the Cauchy-Green deformation tensor determines the FTLE field.

### **Key Features**
- Customizable particle integration strategies.
- Interpolation of velocity fields to particle positions.
- Extensible design supporting multiple file formats.
- Modular and well-structured codebase for easy modifications.

> [!NOTE]
> The current implementation supports MATLAB file formats for input data. However, additional formats can be integrated with minimal effort due to the modular design. The code accepts VTK and MATLAB formats as output.

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
   uv sync
   ```
3. Install src/ directory as an "editable" package within .venv to overcome import issues:
   ```bash
   uv pip install -e '.[dev,test]' --verbose
   ```
   This will make src directory a first-class citizen in the Python environment, which uv respects.

---

## **USAGE**

The code features both a clean, CLI-oriented architecture (utilizing configuration files and
file-based I/O) and a lightweight, notebook-friendly API. The latter allows you to run small-scale
examples entirely in memory, eliminating the need to handle intermediate files, which makes it
perfect for demonstrating in Jupyter notebooks. Several such notebooks, located in the examples
folder, combine analytical velocity fields with visual explanations to illustrate the FTLE
solver’s execution.

> [!TIP]
> For production runs, it is often more practical to read velocity and grid data directly
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
  <img src="https://github.com/las-unicamp/pyFTLE/blob/main/.github/particles.png" alt="Particles Group Image" style="width: 50%; margin-right: 20px;">
</div>


Instead of passing individual MATLAB files directly to the solver, the interface expects a set of
plain text (.txt) files—one for each data type: velocity, coordinate, and particle data. Each of
these .txt files should contain a list of file paths to the corresponding .mat files, with one path
per line. For example, the velocity .txt file will list all the velocity MATLAB files (one per line),
and similarly for the coordinate and particle .txt files. This approach allows the solver to process
sequences of time-resolved data more easily and keeps the input interface clean and scalable.

> [!TIP]
> The `create_list_of_input_files.py` facilitates the creation of these .txt files. 


<a name="anchor-point-running-via-CLI"></a>

### **Running the code via CLI**

The script requires several parameters, which can be passed through the command line or a configuration
file (`config.yaml`) located in the root directory. Among these parameters are .txt files that
indicates the location of the input files in Matlab format (the velocity field, coordinates and
particles).

> [!TIP]
> Once the parameters are properly set, the solver can be executed from the root directory with the following command:
>
> ```bash
> PYTHONPATH=${PWD} uv run python src/main.py -c config.yaml
> ```

Alternatively, you can run the script from the CLI as:

```bash
PYTHONPATH=${PWD} uv run python main.py \
    --experiment_name "my_experiment" \
    --list_velocity_files "velocity_files.txt" \
    --list_grid_files "grid_files.txt" \
    --list_particle_files "particle_files.txt" \
    --snapshot_timestep 0.1 \
    --flow_map_period 5.0 \
    --integrator "rk4" \
    --interpolator "cubic" \
    --num_processes 4 \
    --output_format "vtk" \
    --grid_shape 100,100,100  # comment this line for unstructured data
```

For VSCode users, the script execution can be streamlined via `.vscode/launch.json`.


### **Required Parameters**

| Parameter             | Type    | Description                                                                                   |
| --------------------- | ------- | --------------------------------------------------------------------------------------------- |
| `experiment_name`     | `str`   | Name of the subdirectory where the FTLE fields will be saved.                                 |
| `list_velocity_files` | `str`   | Path to a text file listing velocity data files.                                              |
| `list_grid_files`     | `str`   | Path to a text file listing grid files.                                                       |
| `list_particle_files` | `str`   | Path to a text file listing particle data files.                                              |
| `snapshot_timestep`   | `float` | Timestep between snapshots (positive for forward-time FTLE, negative for backward-time FTLE). |
| `flow_map_period`     | `float` | Integration period for computing the flow map.                                                |
| `integrator`          | `str`   | Time-stepping method (`rk4`, `euler`, `ab2`).                                                 |
| `interpolator`        | `str`   | Interpolation method (`cubic`, `linear`, `nearest`, `grid`).                                  |
| `num_processes`       | `int`   | Number of workers in the multiprocessing pool. Each worker computs the FTLE of a snapshot.    |
| `output_format`       | `str`   | Output format (`mat`, `vtk`).                                                                 |

### **Optional Parameters**

| Parameter             | Type    | Description                                                                                   |
| --------------------- | ------- | --------------------------------------------------------------------------------------------- |
| `grid_shape`          | `int`   | Grid shape for structured points. It must be a comma-separated tuple of integers.             |


---


## **REFERENCES**

A list of scientific works using pyFTLE includes:

- de Souza, Miotto, Wolf. _Active flow control of vertical-axis wind turbines: Insights from large-eddy simulation and finite-time resolvent analysis_. Journal of Fluids and Structures, 2025.
- de Souza, Wolf, Safari, Yeh. _Control of Deep Dynamic Stall by Duty-Cycle Actuation Informed by Stability Analysis_. AIAA Journal, 2025.
- Lui, Wolf. _Interplay between streaks and vortices in shock-boundary layer interactions with conditional bubble events over a turbine airfoil_. Physical Review Fluids, 2025.
- Lui, Wolf, Ricciardi, Gaitonde. _Analysis of Streamwise Vortices in a Supersonic Turbine Cascade_. AIAA Aviation Forum and Ascend, 2024.


---

## **LICENSE**

This project is licensed under the **MIT License**.

---

## **CONTRIBUTING**

When contributing to this repository, please make sure to keep the code well tested.

To run the entire test suit, we recommend the following approach:
```bash
PYTHONPATH=${PWD} uv run python -m pytest
```

---

## **MAIN DEVELOPERS**

- **Renato Fuzaro Miotto**
- **Lucas Feitosa de Souza**
- **William Roberto Wolf**

---

For bug reports, feature requests, or contributions, please open an issue or submit a pull request on [GitHub](https://github.com/las-unicamp/pyFTLE).
