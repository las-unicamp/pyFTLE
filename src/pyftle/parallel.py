"""
Infrastructure Layer

This module provides utilities for parallel task execution, real-time progress
monitoring, and robust error handling in multiprocessing contexts. It is
designed for scientific computations (e.g., FTLE batch processing) where
multiple independent tasks need to be executed concurrently with visual feedback
through progress bars.

Features
--------
- Seamless integration with both terminal and Jupyter environments via `tqdm`.
- Graceful handling of errors during multiprocessing (with traceback display).
- Live monitoring of per-task and global progress using multiple progress bars.
- Clean shutdown and interruption handling across processes.

Classes
-------
ParallelExecutor
    Handles parallel task execution, progress tracking, and error reporting.

Functions
---------
get_tqdm()
    Detects environment and returns the appropriate tqdm class (terminal or notebook).
"""

import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from colorama import Fore, Style
from colorama import init as colorama_init

from pyftle.data_source import BatchSource

colorama_init(autoreset=True)


def get_tqdm():
    """
    Return a tqdm-compatible progress bar for the current environment.

    This function automatically detects whether the code is running inside a
    Jupyter/IPython environment or a standard terminal, and imports the
    appropriate `tqdm` variant accordingly.

    Returns
    -------
    tqdm_class : type
        The appropriate tqdm class:
        - `tqdm.notebook.tqdm` if running in Jupyter/IPython.
        - `tqdm.tqdm` for standard terminal environments.

    Notes
    -----
    This ensures consistent progress bar rendering across environments.
    """
    try:
        from IPython.core.getipython import get_ipython

        shell = get_ipython()
        if shell and hasattr(shell, "config") and "IPKernelApp" in shell.config:
            from tqdm.notebook import tqdm as tqdm_notebook

            return tqdm_notebook
    except Exception:
        pass

    from tqdm import tqdm as tqdm_terminal

    return tqdm_terminal


tqdm = get_tqdm()


class ParallelExecutor:
    """
    Manage multiprocessing execution with live progress monitoring.

    The `ParallelExecutor` is responsible for executing multiple independent
    tasks in parallel using `concurrent.futures.ProcessPoolExecutor`, while
    providing live progress visualization and robust error handling.

    Parameters
    ----------
    n_processes : int, optional
        Number of parallel worker processes to launch (default is 4).

    Attributes
    ----------
    n_processes : int
        Number of concurrent worker processes.
    progress_queue : multiprocessing.Queue
        Shared queue used to communicate task progress between worker processes
        and the monitor process.
    _stop_event : multiprocessing.Event
        Event flag used to signal the monitor process to terminate.
    """

    def __init__(self, n_processes: int = 4):
        self.n_processes = n_processes
        manager = mp.Manager()
        self.progress_queue = manager.Queue()
        self._stop_event = manager.Event()

    def _monitor_progress(self, total_tasks: int, steps_per_task: int):
        """
        Display real-time progress for all parallel tasks.

        This method runs in a dedicated process and continuously listens to the
        shared progress queue. It maintains:
        - One global progress bar tracking total completed tasks.
        - Individual progress bars for active tasks.

        Parameters
        ----------
        total_tasks : int
            Total number of tasks to monitor.
        steps_per_task : int
            Expected number of progress updates (e.g., time steps) per task.

        Notes
        -----
        - Each task reports its progress as `(task_id, step)` tuples to the queue.
        - When a task is complete, it sends `(task_id, "done")`.
        - The method terminates when all tasks are done or `_stop_event` is set.
        """

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
        """
        Execute multiple tasks in parallel and collect results.

        Each task is executed in a separate process via the provided `worker_fn`.
        Progress updates are collected asynchronously and displayed via tqdm bars.

        Parameters
        ----------
        tasks : list of BatchSource
            List of task objects to process. Each `BatchSource` must contain
            attributes such as `id` and `num_steps`, representing the task's
            identity and number of progress steps, respectively.
        worker_fn : callable
            Function with signature `worker_fn(task, queue)` that performs the
            actual work for each task.

            The function must:

            - Report progress by placing `(task.id, step)` or `(task.id, "done")`
              messages into the shared queue.
            - Return a result (e.g., NumPy array) or raise an exception on failure.

        Returns
        -------
        results : list of np.ndarray or None
            List of task results in the same order as the input tasks.

        Raises
        ------
        RuntimeError
            If one or more tasks fail during execution. Errors are printed with
            traceback details, and all remaining tasks are immediately canceled.

        Notes
        -----
        - Progress is displayed live for all tasks.
        - If an error occurs in any task, all ongoing computations are stopped.
        - The method ensures that monitor processes and worker pools terminate
          cleanly even in error conditions.
        """

        steps_per_task = tasks[0].num_steps  # num snapshots in flow map period

        monitor_proc = mp.Process(
            target=self._monitor_progress,
            args=(len(tasks), steps_per_task),
        )
        monitor_proc.start()

        results: list[np.ndarray | None] = [None] * len(tasks)
        exceptions = []

        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            futures = {
                executor.submit(worker_fn, task, self.progress_queue): i
                for i, task in enumerate(tasks)
            }

            for future in as_completed(futures):
                i = futures[future]
                task = tasks[i]
                try:
                    result = future.result()
                    results[i] = result  # preserve task order
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

        return results
