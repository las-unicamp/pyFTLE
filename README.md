# **pyFTLE: A Python Package for Computing Finite-Time Lyapunov Exponents**

`pyFTLE` computes hyperbolic Lagrangian Coherent Structures (LCS) from velocity flow field data using Finite-Time Lyapunov Exponents (FTLE).

---

## **Overview**

<div align="center">
  <img src="https://github.com/las-unicamp/pyFTLE/blob/main/.github/integration.gif" width="45%" />
  <img src="https://github.com/las-unicamp/pyFTLE/blob/main/.github/ftle.gif" width="45%" />
</div>


pyFTLE provides a robust and modular implementation for computing FTLE fields. It tracks particle positions over time by interpolating a given velocity field and integrating their motion. After a specified integration period, the flow map Jacobian is computed, and the largest eigenvalue of the Cauchy-Green deformation tensor determines the FTLE field.

### **Key Features**
- Customizable particle integration strategies.
- Interpolation of velocity fields to particle positions.
- Extensible design supporting multiple file formats.
- Modular and well-structured codebase for easy modifications.

Currently, the package supports MATLAB file formats for input data. However, additional formats can be integrated with minimal effort due to the modular design.

---

## **Installation**

### **Requirements**
- Python 3.12+

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

---

## **Usage**

### **Running the Script**

The script requires several parameters, which can be passed through the command line or a configuration file.

#### **VSCode Users**
If you're using VSCode, you can configure the `.vscode/launch.json` file to streamline script execution.

#### **Command Line Execution**
Run the script with the required parameters:
```bash
python main.py --experiment_name "my_experiment" \
               --list_velocity_files "velocity_files.txt" \
               --list_grid_files "grid_files.txt" \
               --list_particle_files "particle_files.txt" \
               --snapshot_timestep 0.1 \
               --flow_map_period 5.0 \
               --integrator "rk4" \
               --interpolator "cubic" \
               --num_processes 4
```

Alternatively, use a configuration file:
```bash
python main.py -c config.yaml
```

### **Required Parameters**

| Parameter               | Type    | Description                                                                                   |
| ----------------------- | ------- | --------------------------------------------------------------------------------------------- |
| `--experiment_name`     | `str`   | Name of the subdirectory where the FTLE fields will be saved.                                 |
| `--list_velocity_files` | `str`   | Path to a text file listing velocity data files.                                              |
| `--list_grid_files`     | `str`   | Path to a text file listing grid files.                                                       |
| `--list_particle_files` | `str`   | Path to a text file listing particle data files.                                              |
| `--snapshot_timestep`   | `float` | Timestep between snapshots (positive for forward-time FTLE, negative for backward-time FTLE). |
| `--flow_map_period`     | `float` | Integration period for computing the flow map.                                                |
| `--integrator`          | `str`   | Time-stepping method (`rk4`, `euler`, `ab2`).                                                 |
| `--interpolator`        | `str`   | Interpolation method (`cubic`, `linear`, `nearest`, `grid`).                                  |
| `--num_processes`       | `int`   | Number of workers in the multiprocessing pool. Each worker computs the FTLE of a snapshot.    |


### **File Requirements**

- The `list_velocity_files` must be a `.txt` file with the path to the velocity files. Make sure the listed files are ordered according to their simulation time (ascending order).
 
- The `list_grid_files` and `list_particle_files` must also be `.txt` files. For moving bodies, you can have multiple files (same number as the number of velocity files), but if the mesh grid is fixed, then you can have a single item in both `list_grid_files` and `list_particle_files` lists.

- The current implementation supports MATLAB file formats. The MATLAB velocity file should contain columns labeled `velocity_x` and `velocity_y`. The grid file should include `coordinate_x` and `coordinate_y` headers.

- The particle files must include the headers: `left`, `right`, `top`, and `bottom`, as illustrated in the accompanying figure. These headers define the positions of four neighboring particles surrounding a central location, where the FTLE is computed.

<div align="center">
  <img src="https://github.com/las-unicamp/pyFTLE/blob/main/.github/particles.png" alt="Paticles Group Image" style="width: 50%; margin-right: 20px;">
</div>

> **NOTE:** The current implementation supports MATLAB file formats with the mentioned file requirements. However, the user can implement their own readers to accept files with different data structure.

---

## **License**

This project is licensed under the **MIT License**.

---

## **Contributors**

- **Renato Fuzaro Miotto**
- **Lucas Feitosa de Souza**
- **William Roberto Wolf**

---

For bug reports, feature requests, or contributions, please open an issue or submit a pull request on [GitHub](https://github.com/las-unicamp/pyFTLE).

