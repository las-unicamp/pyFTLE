import itertools
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

from colorama import Fore, Style
from colorama import init as colorama_init
from tqdm import tqdm

from src.data_source import BatchSource, FileBatchSource
from src.decorators import timeit
from src.file_utils import get_files_list
from src.file_writers import create_writer
from src.hyperparameters import args
from src.integrate import create_integrator
from src.interpolate import create_interpolator
from src.process import FTLESolver  # Domain layer

colorama_init(autoreset=True)


# ─────────────────────────────────────────────────────────────
# Infrastructure Layer (handles parallelism, tqdm, and errors)
# ─────────────────────────────────────────────────────────────


class ParallelExecutor:
    """Handles multiprocessing and live progress monitoring."""

    def __init__(self, n_processes: int = 4):
        self.n_processes = n_processes
        manager = mp.Manager()
        self.progress_queue = manager.Queue()
        self._stop_event = manager.Event()

    def _monitor_progress(self, total_tasks: int, steps_per_task: int):
        """Monitor progress in a separate process."""
        global_bar = tqdm(
            total=total_tasks, desc="Global", position=0, dynamic_ncols=True
        )
        active_bars = {}
        available_slots = list(range(1, self.n_processes + 1))
        finished = 0

        while not self._stop_event.is_set() and finished < total_tasks:
            while not self.progress_queue.empty():
                task_id, status = self.progress_queue.get()
                if status == "done":
                    if task_id in active_bars:
                        pos = active_bars[task_id].pos
                        active_bars[task_id].close()
                        del active_bars[task_id]
                        available_slots.append(pos)
                    global_bar.update(1)
                    finished += 1
                else:
                    if task_id not in active_bars and available_slots:
                        pos = available_slots.pop(0)
                        bar = tqdm(
                            total=steps_per_task,
                            desc=task_id,
                            position=pos,
                            leave=False,
                            dynamic_ncols=True,
                        )
                        active_bars[task_id] = bar
                    bar = active_bars[task_id]
                    bar.n = status
                    bar.refresh()
            time.sleep(0.05)

        global_bar.close()
        for bar in active_bars.values():
            bar.close()

    def run(self, tasks: list[BatchSource], worker_fn):
        """Run worker_fn(task, queue) in parallel and handle errors."""
        steps_per_task = tasks[0].num_steps  # num snapshots in flow map period

        monitor_proc = mp.Process(
            target=self._monitor_progress,
            args=(len(tasks), steps_per_task),
        )
        monitor_proc.start()

        exceptions = []
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            futures = {
                executor.submit(worker_fn, task, self.progress_queue): task
                for task in tasks
            }

            for future in as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                except Exception as e:
                    error_msg = (
                        f"\n{Fore.RED}❌ Error in task "
                        f"{task.id}:{Style.RESET_ALL}\n"
                        f"{traceback.format_exc()}"
                    )
                    print(error_msg, flush=True)
                    exceptions.append((task, e))

                    # Signal stop immediately
                    self._stop_event.set()

                    # 🔥 Cancel remaining futures and shut down pool immediately
                    executor.shutdown(wait=False, cancel_futures=True)

                    # 🔥 Kill monitor process right away
                    if monitor_proc.is_alive():
                        monitor_proc.terminate()

                    raise  # re-raise to exit as_completed loop immediately

        # Ensure monitor process is dead
        if monitor_proc.is_alive():
            monitor_proc.terminate()
        monitor_proc.join(timeout=0.5)

        if exceptions:
            print(
                f"{Fore.RED}\n⚠️  {len(exceptions)} task(s) failed. "
                "See messages above for details."
                f"{Style.RESET_ALL}",
                flush=True,
            )
            raise RuntimeError("One or more FTLE batches failed.")


# ─────────────────────────────────────────────────────────────
# Application Layer (coordinates domain and infrastructure)
# ─────────────────────────────────────────────────────────────


class MultipleFTLEProcessManager:
    """Reads snapshot list, creates batches, and runs FTLE solvers in parallel."""

    def __init__(self):
        self.snapshot_files: List[str] = get_files_list(args.list_velocity_files)
        self.coordinate_files: List[str] = get_files_list(args.list_coordinate_files)
        self.particle_files: List[str] = get_files_list(args.list_particle_files)
        self.timestep: float = args.snapshot_timestep
        self.executor = ParallelExecutor(n_processes=args.num_processes)
        self.grid_shape = args.grid_shape

        interpolator = create_interpolator(args.interpolator)

        self.integrator = create_integrator(args.integrator, interpolator)

        output_dir = os.path.join("outputs", args.experiment_name)
        self.writer = create_writer(args.output_format, output_dir, args.grid_shape)

        self._handle_time_direction()

    def _handle_time_direction(self) -> None:
        """Handles time direction for backward/forward FTLE computation."""
        if self.timestep < 0:
            self.snapshot_files.reverse()
            self.coordinate_files.reverse()
            self.particle_files.reverse()
            print("Running backward-time FTLE")
        else:
            print("Running forward-time FTLE")

    def _create_batches(self) -> list[BatchSource]:
        """Generate overlapping batches of snapshot, coordinate and particle files"""
        num_snapshots_total = len(self.snapshot_files)
        num_snapshots_in_flow_map_period = (
            int(args.flow_map_period / abs(args.snapshot_timestep)) + 1
        )

        p = num_snapshots_in_flow_map_period
        n = num_snapshots_total

        # Precompute snapshot file batches
        snapshot_batches = [self.snapshot_files[i : i + p] for i in range(n - p + 1)]

        # Precompute coordinate file batches (cycled)
        coord_cycle = list(
            itertools.islice(itertools.cycle(self.coordinate_files), n + p - 1)
        )
        coordinate_batches = [coord_cycle[i : i + p] for i in range(n - p + 1)]

        # Precompute particle file selection (cycled)
        particle_cycle = list(itertools.islice(itertools.cycle(self.particle_files), n))
        particle_batches = [particle_cycle[i] for i in range(n - p + 1)]  # pick one str

        tasks: list[BatchSource] = []

        for i in range(n - p + 1):
            task = FileBatchSource(
                snapshot_files=list(snapshot_batches[i]),
                coordinate_files=list(coordinate_batches[i]),
                particle_file=particle_batches[i],  # Assume single particle file
                snapshot_timestep=self.timestep,
                flow_map_period=p,
                grid_shape=self.grid_shape,
            )
            tasks.append(task)
        return tasks

    def _worker(self, batch_source: BatchSource, progress_queue):
        """Wrapper function for parallel execution."""
        solver = FTLESolver(
            batch_source,
            integrator=self.integrator,
            progress_queue=progress_queue,
            output_writer=self.writer,
        )
        solver.run()

    def run(self):
        batches = self._create_batches()
        self.executor.run(batches, self._worker)


# ─────────────────────────────────────────────────────────────
# Usage Example
# ─────────────────────────────────────────────────────────────


@timeit
def main():
    try:
        manager = MultipleFTLEProcessManager()
        manager.run()
    except RuntimeError as e:
        print(f"{Fore.RED}\n❌ Execution stopped: {e}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
