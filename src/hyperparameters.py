import os
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
    input_directory: str
    snapshot_file_format: str
    snapshot_timestep: float
    particles_filename: str
    flow_map_period: float
    grid_filename: str


def directory(raw_path: str) -> str:
    if not os.path.isdir(raw_path):
        raise configargparse.ArgumentTypeError(
            f'"{raw_path}" is not an existing directory. '
            "Make sure to create it before running the code."
        )
    return os.path.abspath(raw_path)


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
    "--input_directory",
    type=directory,
    default=os.path.curdir,
    help="Directory to search for snapshots. default='working directory'",
)
parser.add_argument(
    "--snapshot_file_format",
    type=str,
    choices=[".mat", "cgns"],
    default=".mat",
    help="Extension of the snapshots to look for. The user must guarantee that "
    "there exist a proper implementation of the reader for the desired file format. "
    "default=.mat",
)
parser.add_argument(
    "--snapshot_timestep",
    type=float,
    required=True,
    help="Timestep between snapshots.",
)
parser.add_argument(
    "--particles_filename",
    type=str,
    default="particles.csv",
    help="Name of file containing the particles coordinates in csv format. The "
    "column headers must be `coordinate_x` and `coordinate_y`.",
)
parser.add_argument(
    "--flow_map_period",
    type=float,
    required=True,
    help="Approximate period of integration to evaluate the flow map. This value "
    "will be devided by the `snapshot_timestep` to get the number of snapshots.",
)
parser.add_argument(
    "--grid_filename",
    type=str,
    required=True,
    help="Name of gridfile to search for. For the case of moving meshes, provide "
    "a pattern for the name of the grid files to search for. Make sure that all grid "
    "files are located in the `input_directory`.",
)


args = MyProgramArgs(**vars(parser.parse_args()))
