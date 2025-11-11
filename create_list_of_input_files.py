import os

from pyftle.file_utils import find_files_with_pattern, write_list_to_txt

ROOT_DIR = "./inputs/double_gyre"
SAVE_TXT_DIR = "./inputs"


def main():
    # Find all .mat files
    all_mat_files = find_files_with_pattern(ROOT_DIR, ".mat")

    # Separate velocity files, coordinate and particle file(s)
    velocity_files = sorted([f for f in all_mat_files if "velocities" in f])
    coordinate_files = sorted([f for f in all_mat_files if "coordinate" in f])
    particle_files = sorted([f for f in all_mat_files if "particles" in f])

    if not velocity_files:
        print("Warning: No velocity files found.")
    if not coordinate_files:
        print("Warning: No coordinate file found.")
    if not coordinate_files:
        print("Warning: No particle file found.")

    # Write velocity file list
    velocity_txt_path = os.path.join(SAVE_TXT_DIR, "inputs_velocity.txt")
    write_list_to_txt(velocity_files, velocity_txt_path)
    print(f"Velocity file list saved to: {velocity_txt_path}")

    # Write coordinate file list (single file expected)
    coordinate_txt_path = os.path.join(SAVE_TXT_DIR, "inputs_coordinate.txt")
    write_list_to_txt(coordinate_files, coordinate_txt_path)
    print(f"Coordinate file list saved to: {coordinate_txt_path}")

    # Write particle file list (single file expected)
    particle_txt_path = os.path.join(SAVE_TXT_DIR, "inputs_particle.txt")
    write_list_to_txt(particle_files, particle_txt_path)
    print(f"Particle file list saved to: {particle_txt_path}")


if __name__ == "__main__":
    main()
