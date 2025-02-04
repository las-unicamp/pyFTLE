import os

from src.file_utils import find_files_with_pattern, write_list_to_txt

ROOT_DIR = "./inputs/double_gyre"
SAVE_TXT_DIR = "./inputs"


def main():
    # Find all .mat files
    all_mat_files = find_files_with_pattern(ROOT_DIR, ".mat")

    # Separate velocity files and grid file(s)
    velocity_files = sorted([f for f in all_mat_files if "velocities" in f])
    grid_files = sorted(
        [f for f in all_mat_files if "grid" in f]
    )  # Assuming a single grid file
    particle_files = sorted(
        [f for f in all_mat_files if "particles" in f]
    )  # Assuming a single grid file

    if not velocity_files:
        print("Warning: No velocity files found.")
    if not grid_files:
        print("Warning: No grid file found.")
    if not grid_files:
        print("Warning: No particle file found.")

    # Write velocity file list
    velocity_txt_path = os.path.join(SAVE_TXT_DIR, "inputs_velocity.txt")
    write_list_to_txt(velocity_files, velocity_txt_path)
    print(f"Velocity file list saved to: {velocity_txt_path}")

    # Write grid file list (single file expected)
    grid_txt_path = os.path.join(SAVE_TXT_DIR, "inputs_grid.txt")
    write_list_to_txt(grid_files, grid_txt_path)
    print(f"Grid file list saved to: {grid_txt_path}")

    # Write particle file list (single file expected)
    particle_txt_path = os.path.join(SAVE_TXT_DIR, "inputs_particle.txt")
    write_list_to_txt(particle_files, particle_txt_path)
    print(f"Particle file list saved to: {particle_txt_path}")


if __name__ == "__main__":
    main()
