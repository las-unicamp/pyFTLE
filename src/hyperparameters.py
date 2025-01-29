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
    logging_root: str
    experiment_name: str

    # input parameters
    list_velocity_files: str
    list_grid_files: str
    particles_filename: str
    snapshot_timestep: float
    flow_map_period: float


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
    "--logging_root", type=str, default="./logs", help="Root for logging"
)
parser.add_argument(
    "--experiment_name",
    type=str,
    required=True,
    help="Name of subdirectory in logging_root where summaries and checkpoints"
    "will be saved.",
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
    "--particles_filename",
    type=str,
    default="particles.csv",
    help="Name of file containing the particles coordinates in csv format. The "
    "column headers must be `coordinate_x` and `coordinate_y`.",
)
parser.add_argument(
    "--snapshot_timestep",
    type=float,
    required=True,
    help="Timestep between snapshots.",
)
parser.add_argument(
    "--flow_map_period",
    type=float,
    required=True,
    help="Approximate period of integration to evaluate the flow map. This value "
    "will be devided by the `snapshot_timestep` to get the number of snapshots.",
)


args = MyProgramArgs(**vars(parser.parse_args()))
