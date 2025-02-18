from dataclasses import dataclass

import configargparse


@dataclass
class MyProgramArgs:
    """
    This is a helper to provide typehints of the arguments.
    All possible arguments must be declared in this dataclass.
    """

    config_filepath: any

    # logger parameters
    experiment_name: str

    # input parameters
    list_velocity_files: str
    list_grid_files: str
    list_particle_files: str
    snapshot_timestep: float
    flow_map_period: float
    integrator: str
    interpolator: str
    num_processes: int


parser = configargparse.ArgumentParser()


parser.add(
    "-c",
    "--config_filepath",
    required=False,
    is_config_file=True,
    help="Path to config file.",
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
    "--list_grid_files",
    type=str,
    required=True,
    help="Text file containing a list (columnwise) of paths to grid files. "
    "The user must guarantee that there exist a proper implementation of the "
    "reader for the desired grid file format.",
)
parser.add_argument(
    "--list_particle_files",
    type=str,
    required=True,
    help="Text file containing a list (columnwise) of paths to particle files. "
    "Each file must contain headers `left`, `right`, `top` and `bottom` to "
    "help identify the group of particles to evaluate the Cauchy-Green deformation "
    "tensor. The user must guarantee that there exist a proper implementation of the "
    "reader for the desired grid file format.",
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
    "will be devided by the `snapshot_timestep` to get the number of snapshots.",
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


args = MyProgramArgs(**vars(parser.parse_args()))
