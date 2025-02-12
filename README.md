# **pyLCS: A Python Package for Computing Finite-Time Lyapunov Exponents**

`pyLCS` computes hyperbolic Lagrangian Coherent Structures (LCS) from velocity flow field data using Finite-Time Lyapunov Exponents (FTLE).

---

## **Overview**

<p align="center">
  <img src="https://github.com/las-unicamp/pyLCS/blob/main/.github/integration.gif" width="45%" />
  <img src="https://github.com/las-unicamp/pyLCS/blob/main/.github/ftle.gif" width="45%" />
</p>


This package provides a standard implementation for computing the FTLE field. It tracks particle positions over time by interpolating a given velocity field and integrating their motion. After a specified integration period, the flow map Jacobian is computed, and the largest eigenvalue of the Cauchy-Green deformation tensor is used to determine the FTLE field.

Key features include:
- Customizable particle integration strategies.
- Interpolation of velocity fields to particle positions.
- Extensibility to support multiple file formats.
- A modular and well-structured codebase for easy modifications.

Currently, the package supports MATLAB file formats for input data. However, additional formats can be integrated with minimal effort due to the modular design.

---

## **Installation**

### **Requirements**
- Python 3.12+

> ⚠️ **macOS and Windows users:** CUDA-enabled builds are only supported on Linux. To use the package on macOS or Windows, refer to the [UV-PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/#using-a-pytorch-index).

### **Using UV (Recommended)**

[UV](https://docs.astral.sh/uv/) is a modern Python package and project manager that simplifies dependency management.

#### **Installation Steps:**
1. Clone the repository:
   ```bash
   git clone https://github.com/las-unicamp/measuring_forces_on_sand_dunes.git
   cd measuring_forces_on_sand_dunes
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
If you're using VSCode, you can configure the `.vscode/launch.json` file for running the script conveniently.

#### **Command Line Execution**
Run the script with the required parameters:
```bash
python main.py --experiment_name "my_experiment" \
               --list_velocity_files "velocity_files.txt" \
               --list_grid_files "grid_files.txt" \
               --list_particle_files "particle_files.txt" \
               --snapshot_timestep 0.1 \
               --flow_map_period 5.0 \
               --integrator "rk4"
```

Alternatively, use a configuration file:
```bash
python main.py -c config.yaml
```

### **Required Parameters**

| Parameter               | Type    | Description                                                                                   |
| ----------------------- | ------- | --------------------------------------------------------------------------------------------- |
| `--logging_root`        | `str`   | Directory for storing logs (default: `./logs`).                                               |
| `--experiment_name`     | `str`   | Name of the subdirectory for logs and checkpoints.                                            |
| `--list_velocity_files` | `str`   | Path to a text file listing velocity data files.                                              |
| `--list_grid_files`     | `str`   | Path to a text file listing grid files.                                                       |
| `--list_particle_files` | `str`   | Path to a text file listing particle data files.                                              |
| `--snapshot_timestep`   | `float` | Timestep between snapshots (positive for forward-time FTLE, negative for backward-time FTLE). |
| `--flow_map_period`     | `float` | Integration period for computing the flow map.                                                |
| `--integrator`          | `str`   | Time-stepping method (`rk4`, `euler`, `ab2`).                                                 |

---

## **License**

This project is licensed under the **MIT License**.

---

## **Contributors**

- **Renato Fuzaro Miotto**
- **Lucas Feitosa de Souza**
- **William Roberto Wolf**

---

For bug reports, feature requests, or contributions, please open an issue or submit a pull request on [GitHub](https://github.com/las-unicamp/pyLCS).

