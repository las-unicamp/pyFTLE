# ruff: noqa: N803, N806
import os

import numpy as np
from matplotlib import pyplot as plt
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

    print(f"Timestep: {times[1] - times[0]}")

    # Save grid file test loading it separetely
    data_dict = {
        "coordinate_x": coords[:, 0],
        "coordinate_y": coords[:, 1],
    }
    filename = os.path.join(output_dir, "grid.mat")
    savemat(filename, data_dict)
    print(f"Saved {filename}")

    # Create a seed of neighboring particles

    spacing = 0.025  # small gap for neighboring particles
    x_min, x_max = 0, 2
    y_min, y_max = 0, 1

    # Define the valid region for central locations (avoiding boundaries)
    margin = 2 * spacing
    x_valid_min, x_valid_max = x_min + margin, x_max - margin
    y_valid_min, y_valid_max = y_min + margin, y_max - margin

    # num_particles = 100
    # rng = np.random.default_rng(42)  # For reproducibility
    # central_x = rng.uniform(x_valid_min, x_valid_max, num_particles)
    # central_y = rng.uniform(y_valid_min, y_valid_max, num_particles)
    # central_locations = np.column_stack((central_x, central_y))

    approximate_num_particles = 100
    num_x = int(
        np.sqrt(approximate_num_particles * (x_max - x_min) / (y_max - y_min))
    )  # Adjust based on aspect ratio
    num_y = int(approximate_num_particles / num_x)
    central_x = np.linspace(x_valid_min, x_valid_max, num_x)
    central_y = np.linspace(y_valid_min, y_valid_max, num_y)
    Xc, Yc = np.meshgrid(central_x, central_y)
    central_locations = np.column_stack((Xc.ravel(), Yc.ravel()))

    coordinates_top = central_locations + np.array([0, spacing])
    coordinates_bottom = central_locations - np.array([0, spacing])
    coordinates_left = central_locations - np.array([spacing, 0])
    coordinates_right = central_locations + np.array([spacing, 0])

    particles = {
        "top": coordinates_top,
        "bottom": coordinates_bottom,
        "left": coordinates_left,
        "right": coordinates_right,
    }
    filename = os.path.join("./", "particles.mat")
    savemat(filename, particles)
    print(f"Saved {filename}")

    particles_plot = {
        "top": (coordinates_top, "red", "Top"),
        "bottom": (coordinates_bottom, "blue", "Bottom"),
        "left": (coordinates_left, "green", "Left"),
        "right": (coordinates_right, "purple", "Right"),
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("Particle Groups Inside the Domain")
    for key, (coords, color, label) in particles_plot.items():
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            color=color,
            label=label,
            alpha=0.7,
            edgecolors="black",
        )

    # Scatter plot for central locations (black)
    ax.scatter(
        central_locations[:, 0],
        central_locations[:, 1],
        color="black",
        label="Central",
        marker="x",
    )
    ax.legend()
    plt.savefig("particles.png")
