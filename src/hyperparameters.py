from dataclasses import dataclass

import configargparse  # type: ignore


@dataclass
class MyProgramArgs:
    """
    This is a helper to provide typehints of the arguments.
    All possible arguments must be declared in this dataclass.
    """

    # logger parameters
    experiment_name: str

    # input parameters
    list_velocity_files: str
    list_coordinate_files: str
    list_particle_files: str
    snapshot_timestep: float
    flow_map_period: float
    integrator: str
    interpolator: str
    num_processes: int

    # configuration
    output_format: str
    grid_shape: tuple[int, ...]


parser = configargparse.ArgumentParser()


def parse_tuple(value: str):
    """
    Convert a string of integers separated by commas into a tuple of integers
    """
    return tuple(map(int, value.split(",")))


# YAML configuration
parser.add_argument(
    "-c",
    "--config",
    is_config_file=True,
    help="Path to configuration file in YAML format",
)

# logger parameters
parser.add_argument(
    "--experiment_name",
    type=str,
    required=True,
    help="Name of subdirectory in root_dir/outputs/ where the outputs will be saved",
)


# input parameters
parser.add_argument(
    "--list_velocity_files",
    type=str,
    required=True,
    help="Text file containing a list (columnwise) of paths to velocity files. "
    "The user must guarantee that there exist a proper implementation of the "
    "reader for the desired velocity file format.",
)
parser.add_argument(
    "--list_coordinate_files",
    type=str,
    required=True,
    help="Text file containing a list (columnwise) of paths to coordinate files. "
    "The user must guarantee that there exist a proper implementation of the "
    "reader for the desired file format.",
)
parser.add_argument(
    "--list_particle_files",
    type=str,
    required=True,
    help="Text file containing a list (columnwise) of paths to particle files. "
    "Each file must contain headers `left`, `right`, `top` and `bottom` to "
    "help identify the group of particles to evaluate the Cauchy-Green deformation "
    "tensor. The user must guarantee that there exist a proper implementation of the "
    "reader for the desired file format.",
)
parser.add_argument(
    "--snapshot_timestep",
    type=float,
    required=True,
    help="Timestep between snapshots. If positive, the forward-time FTLE field "
    "is computed. If negative, then the backward-time FTLE is computed.",
)
parser.add_argument(
    "--flow_map_period",
    type=float,
    required=True,
    help="Approximate period of integration to evaluate the flow map. This value "
    "will be divided by the `snapshot_timestep` to get the number of snapshots.",
)
parser.add_argument(
    "--integrator",
    type=str,
    choices=["rk4", "euler", "ab2"],
    help="Select the time-stepping method to integrate the particles in time. "
    "default='euler'",
)
parser.add_argument(
    "--interpolator",
    type=str,
    choices=["cubic", "linear", "nearest", "grid", "grid_cython"],
    help="Select interpolator strategy to evaluate the particle velocity at "
    "their current location. default='cubic'",
)
parser.add_argument(
    "--num_processes",
    type=int,
    default=1,
    help="Number of workers in the multiprocessing pool. Each worker will compute "
    "the FTLE field of a given snapshot. default=1 (no parallelization)",
)

parser.add_argument(
    "--output_format",
    type=str,
    choices=["mat", "vtk"],
    help="Select output file format. default='mat'",
)
parser.add_argument(
    "--grid_shape",
    type=parse_tuple,
    help="Leverage grid structure of data to efficiently save output files. "
    "Must be passed as a tuple of integers, e.g., --grid_shape 10,10,10 "
    "Leave empty for unstructured point distribution (default).",
)

raw_args = vars(parser.parse_args())
raw_args.pop("config", None)

args = MyProgramArgs(**raw_args)
