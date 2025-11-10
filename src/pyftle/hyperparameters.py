from dataclasses import dataclass

import configargparse  # type: ignore


@dataclass
class MyProgramArgs:
    """
    Dataclass defining the full set of configuration parameters for the FTLE
    computation program.

    This class provides type hints for all arguments parsed by the command-line
    interface (CLI) or YAML configuration file. It ensures that every possible
    argument is declared explicitly, simplifying IDE autocompletion and static
    type checking.

    Attributes
    ----------
    experiment_name : str
        Name of the subdirectory inside `root_dir/outputs/` where the output
        files will be stored.

    list_velocity_files : str
        Path to a text file containing a list (one per line) of velocity data
        file paths. The user must ensure a compatible reader exists for the
        chosen format.

    list_coordinate_files : str
        Path to a text file containing a list (one per line) of coordinate
        file paths. The user must ensure a compatible reader exists for the
        chosen format.

    list_particle_files : str
        Path to a text file containing a list (one per line) of particle
        file paths. Each file must contain headers such as `left`, `right`,
        `top`, and `bottom` that identify groups of neighboring particles used
        to compute the Cauchy–Green deformation tensor.

    snapshot_timestep : float
        Time interval between consecutive snapshots. A positive value computes
        forward-time FTLE; a negative value computes backward-time FTLE.

    flow_map_period : float
        Approximate time period over which the flow map is evaluated. The
        integration length (in number of snapshots) is computed as
        `flow_map_period / snapshot_timestep`.

    integrator : str
        Name of the time-integration scheme. Options are:
        `'rk4'`, `'euler'`, or `'ab2'`.

    interpolator : str
        Name of the interpolation strategy to evaluate particle velocity at
        arbitrary positions. Options are: `'cubic'`, `'linear'`, `'nearest'`,
        or `'grid'`.

    num_processes : int
        Number of parallel worker processes used for multiprocessing. Each
        process computes the FTLE field for one snapshot. Default is 1
        (serial execution).

    output_format : str
        Output file format. Options are `'mat'` or `'vtk'`.

    flow_grid_shape : tuple[int, ...]
        Shape of the regular grid used to interpolate flow fields. Must contain
        2 or 3 integers. If empty, the code assumes unstructured data.

    particles_grid_shape : tuple[int, ...]
        Shape of the regular grid used to save particle-based results. Must
        contain 2 or 3 integers. If empty, the code assumes unstructured data.
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
    flow_grid_shape: tuple[int, ...]
    particles_grid_shape: tuple[int, ...]


parser = configargparse.ArgumentParser()


def parse_tuple(value: str):
    """
    Convert a comma-separated string into a tuple of positive integers.

    This function is used to parse command-line arguments representing grid
    shapes (e.g., `--grid_shape 10,20,30`). It validates that the input contains
    either two or three positive integers. If the last integer equals `1`, it
    is automatically removed to handle degenerate dimensions.

    Parameters
    ----------
    value : str
        A string of comma-separated integers, such as `"10,20,30"` or `"32,32"`.

    Returns
    -------
    tuple[int, ...]
        A tuple of two or three positive integers. If the last element is `1`,
        it is omitted.

    Raises
    ------
    ValueError
        If the input does not contain exactly 2 or 3 integers, or if any
        integer is non-positive.

    Examples
    --------
    >>> parse_tuple("10,20,30")
    (10, 20, 30)

    >>> parse_tuple("10,20,1")
    (10, 20)
    """
    # Convert string to a list of integers
    parsed_values = list(map(int, value.split(",")))

    # Check if there are either 2 or 3 elements
    if len(parsed_values) not in [2, 3]:
        raise ValueError("Tuple must contain either 2 or 3 elements.")

    # Check if all integers are positive
    if not all(x > 0 for x in parsed_values):
        raise ValueError("All elements must be positive integers.")

    # If the last element is 1, remove it
    if parsed_values[-1] == 1:
        parsed_values.pop()

    return tuple(parsed_values)


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
    choices=["cubic", "linear", "nearest", "grid"],
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
    "--flow_grid_shape",
    type=parse_tuple,
    help="Leverage grid structure of data to efficiently interpolate. "
    "Must be passed as a tuple of integers, e.g., --grid_shape 10,10,10 "
    "Leave empty for unstructured point distribution (default).",
)
parser.add_argument(
    "--particles_grid_shape",
    type=parse_tuple,
    help="Leverage grid structure of data to efficiently save output files. "
    "Must be passed as a tuple of integers, e.g., --grid_shape 10,10,10 "
    "Leave empty for unstructured point distribution (default).",
)


def parse_args() -> MyProgramArgs:
    """
    Parse command-line arguments and return a `MyProgramArgs` dataclass instance.
    This function should be called explicitly by the main entry point, so that
    importing this module does not trigger parsing automatically.
    """
    raw_args = vars(parser.parse_args())
    raw_args.pop("config", None)
    return MyProgramArgs(**raw_args)
