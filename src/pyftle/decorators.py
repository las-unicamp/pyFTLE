from datetime import datetime
from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def date_diff_in_seconds(dt2: datetime, dt1: datetime) -> int:
    """
    Compute the time difference between two datetime objects in seconds.

    Parameters
    ----------
    dt2 : datetime
        The later datetime.
    dt1 : datetime
        The earlier datetime.

    Returns
    -------
    int
        The total time difference in seconds between `dt2` and `dt1`.

    Examples
    --------
    >>> from datetime import datetime
    >>> t1 = datetime(2024, 1, 1, 12, 0, 0)
    >>> t2 = datetime(2024, 1, 1, 12, 30, 0)
    >>> date_diff_in_seconds(t2, t1)
    1800
    """
    timedelta = dt2 - dt1
    return timedelta.days * 24 * 3600 + timedelta.seconds


def dhms_from_seconds(seconds: int) -> tuple[int, int, int, int]:
    """
    Convert a time duration in seconds into days, hours, minutes, and seconds.

    Parameters
    ----------
    seconds : int
        Duration in seconds.

    Returns
    -------
    tuple of int
        A tuple `(days, hours, minutes, seconds)` representing the same duration.

    Examples
    --------
    >>> dhms_from_seconds(90061)
    (1, 1, 1, 1)
    """
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return days, hours, minutes, seconds


def time_it(func: F) -> F:
    """
    Decorator that measures and prints the execution time of a function.

    The execution time is displayed in days, hours, minutes, and seconds.

    Parameters
    ----------
    func : Callable
        The function to be wrapped and timed.

    Returns
    -------
    Callable
        The wrapped function that prints execution duration upon completion.

    Examples
    --------
    >>> @time_it
    ... def slow_function():
    ...     import time; time.sleep(2)
    ...
    >>> slow_function()
    Execution complete in 0 days, 0 hours, 0 minutes, 2 seconds
    """

    @wraps(func)
    def timeit_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        elapsed_time = date_diff_in_seconds(end_time, start_time)
        days, hours, minutes, seconds = dhms_from_seconds(elapsed_time)
        print(
            "Execution complete in "
            + f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds"
        )
        return result

    return timeit_wrapper  # type: ignore
