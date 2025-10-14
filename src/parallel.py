"""Infrastructure Layer (handles parallelism, tqdm, and errors)"""

import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from colorama import Fore, Style
from colorama import init as colorama_init
from tqdm import tqdm

from src.data_source import BatchSource

colorama_init(autoreset=True)


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
