import itertools
import multiprocessing
import os
import time
from multiprocessing.managers import DictProxy
from multiprocessing.pool import ApplyResult
from typing import Any, Dict, List, cast

from tqdm import tqdm

from src.decorators import timeit
from src.file_readers import (
    CoordinateMatReader,
    VelocityMatReader,
)
from src.file_utils import get_files_list
from src.file_writers import create_writer
from src.hyperparameters import args
from src.interpolate import InterpolatorFactory
from src.process import SnapshotProcessor


class FTLEComputationManager:
    """Manages the distribution of snapshot processing tasks."""

    def __init__(self):
        self.snapshot_files: List[str] = get_files_list(args.list_velocity_files)
        self.grid_files: List[str] = get_files_list(args.list_grid_files)
        self.particle_files: List[str] = get_files_list(args.list_particle_files)
        self._validate_input_lists()

        self.num_snapshots_total = len(self.snapshot_files)
        self.num_snapshots_in_flow_map_period = (
            int(args.flow_map_period / abs(args.snapshot_timestep)) + 1
        )
        self.num_processes = args.num_processes

        self._handle_time_direction()

    def _validate_input_lists(self):
        """Ensures input lists are correctly formatted."""
        if len(self.grid_files) > 1:
            assert len(self.snapshot_files) == len(self.grid_files)
        if len(self.particle_files) > 1:
            assert len(self.snapshot_files) == len(self.particle_files)

    def _handle_time_direction(self):
        """Handles time direction for backward/forward FTLE computation."""
        if args.snapshot_timestep < 0:
            self.snapshot_files.reverse()
            self.grid_files.reverse()
            self.particle_files.reverse()
            print("Running backward-time FTLE")
        else:
            print("Running forward-time FTLE")

    def run(self) -> None:
        """Runs FTLE computation using multiprocessing with shared progress tracking."""
        pool = multiprocessing.Pool(processes=self.num_processes)
        manager = multiprocessing.Manager()
        progress_dict: DictProxy[int, bool] = manager.dict()
        progress_dict_typed: Dict[int, bool] = cast(Dict[int, bool], progress_dict)

        tqdm_position_queue = manager.Queue()

        # Initialize available tqdm positions (from 1 to num_processes)
        for i in range(1, self.num_processes + 1):
            tqdm_position_queue.put(i)

        tqdm_outer = tqdm(
            total=self.num_snapshots_total - self.num_snapshots_in_flow_map_period + 1,
            desc="Total Progress",
            position=0,
            leave=True,
        )

        interpolator_factory = InterpolatorFactory(
            CoordinateMatReader(), VelocityMatReader()
        )

        is_map_period_invalid = (
            self.num_snapshots_total <= self.num_snapshots_in_flow_map_period + 1
        )
        if is_map_period_invalid:
            raise ValueError("Flow map period is too big")

        output_dir = os.path.join("outputs", args.experiment_name)
        writer = create_writer(args.output_format, output_dir, args.grid_shape)

        tasks: List[ApplyResult[None]] = []
        for i in range(
            self.num_snapshots_total - self.num_snapshots_in_flow_map_period + 1
        ):
            snapshot_files_period = self.snapshot_files[
                i : i + self.num_snapshots_in_flow_map_period
            ]
            grid_files_period = list(
                itertools.islice(
                    itertools.cycle(self.grid_files),
                    i,
                    i + self.num_snapshots_in_flow_map_period,
                )
            )
            particle_file = list(
                itertools.islice(
                    itertools.cycle(self.particle_files), self.num_snapshots_total
                )
            )[i]

            progress_dict[i] = False  # Mark as incomplete

            processor = SnapshotProcessor(
                i,
                snapshot_files_period,
                grid_files_period,
                particle_file,
                tqdm_position_queue,
                progress_dict,
                interpolator_factory,
                writer,
            )
            tasks.append(pool.apply_async(processor.run))

        self._monitor_progress(tasks, progress_dict_typed, tqdm_outer)

        pool.close()
        pool.join()
        tqdm_outer.close()

    def _monitor_progress(
        self,
        tasks: List[ApplyResult[None]],
        tqdm_dict: Dict[int, bool],
        tqdm_outer: Any,
    ) -> None:
        """Monitors the completion of parallel tasks and updates the progress bar."""
        completed = 0
        while completed < len(tasks):
            completed = sum(1 for v in tqdm_dict.values() if v)  # Count completed tasks
            tqdm_outer.update(completed - tqdm_outer.n)  # Increment new completions
            tqdm_outer.refresh()
            time.sleep(2.0)  # Prevents excessive polling, keeping CPU usage low

            # OBS: the computations inside the loop runs asynchronously, thus
            # time.sleep will not held the computations. The present method just
            # waits for notifications of completed tasks.


@timeit
def main():
    """Main execution entry point."""
    manager = FTLEComputationManager()
    manager.run()


if __name__ == "__main__":
    main()
