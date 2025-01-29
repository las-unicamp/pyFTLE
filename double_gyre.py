# ruff: noqa: N803, N806
import os

import numpy as np
from scipy.io import savemat


def double_gyre(nx=100, ny=50, t=0, A=0.1, epsilon=0.25, omega=2 * np.pi / 10):
    """
    Compute the coordinates and velocity field (u, v) for the double-gyre problem.

    Parameters:
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
        t (float): Time at which to compute the velocity field.
        A (float): Amplitude of the flow.
        epsilon (float): Perturbation strength.
        omega (float): Frequency of oscillation.

    Returns:
        coords (np.ndarray): Array of shape (nx * ny, 2) containing (x, y) coordinates.
        velocity (np.ndarray): Array of shape (nx * ny, 2) containing (u, v) velocities.
    """
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    # Compute f(x, t) and its derivative
    f = epsilon * np.sin(omega * t) * X**2 + (1 - 2 * epsilon * np.sin(omega * t)) * X
    df_dx = 2 * epsilon * np.sin(omega * t) * X + (1 - 2 * epsilon * np.sin(omega * t))

    # Compute velocity field
    U = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * Y)
    V = np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * Y) * df_dx

    # Reshape into lists of (x, y) and (u, v)
    coords = np.column_stack((X.ravel(), Y.ravel()))
    velocity = np.column_stack((U.ravel(), V.ravel()))

    return coords, velocity


if __name__ == "__main__":
    output_dir = "./inputs/double_gyre/"
    os.makedirs(output_dir, exist_ok=True)

    times = np.linspace(0, 5, 501)
    for i, t in enumerate(times):
        coords, velocity = double_gyre(nx=100, ny=50, t=t)

        data_dict = {
            "coordinate_x": coords[:, 0],
            "coordinate_y": coords[:, 1],
            "velocity_x": velocity[:, 0],
            "velocity_y": velocity[:, 1],
        }

        filename = os.path.join(output_dir, f"velocities{i:04d}.mat")
        savemat(filename, data_dict)
        print(f"Saved {filename}")

    # Save grid file test loading it separetely
    data_dict = {
        "coordinate_x": coords[:, 0],
        "coordinate_y": coords[:, 1],
    }
    filename = os.path.join(output_dir, "grid.mat")
    savemat(filename, data_dict)
    print(f"Saved {filename}")

    print(f"Timestep: {times[1] - times[0]}")
