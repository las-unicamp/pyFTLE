from datetime import datetime
from functools import wraps
from typing import Any, Callable, TypeVar

from src.particle_position import PositionDict

F = TypeVar("F", bound=Callable[..., Any])


def format_position_dict(func: F) -> F:
    """
    Decorator to handle conversion between PositionDict and NumPy arrays
    for integration functions.

    It ensures:
    - Inputs of type PositionDict are converted to NumPy arrays before integration.
    - The function output is converted back to PositionDict.

    Supports:
    - Euler
    - Runge-Kutta 4
    - Adams-Bashforth 2 (which takes two PositionDict arguments)

    Args:
        func (F): Integration function.

    Returns:
        F: Wrapped function with automatic PositionDict handling.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Detect if the function is a method (i.e., first argument is `self` or `cls`)
        self_or_cls = None
        if isinstance(args[0], (object, type)):  # If first arg is an instance or class
            self_or_cls, *args = args  # Extract `self` or `cls`

        # Extract arguments correctly
        h, y_n, *rest = args  # First arg is h, second is y_n

        # Ensure y_n is a PositionDict
        if not isinstance(y_n, PositionDict):
            raise TypeError(f"Expected PositionDict for y_n, got {type(y_n)}")

        # Convert y_n to an array
        y_n_array = y_n.to_array()

        # Check if we have another PositionDict (Adams-Bashforth 2 case)
        y_np1_array = None
        if rest and isinstance(rest[0], PositionDict):
            y_np1_array = rest[0].to_array()

        # Call the function with formatted arguments
        if y_np1_array is not None:
            result_array = (
                func(self_or_cls, h, y_n_array, y_np1_array, *rest[1:], **kwargs)
                if self_or_cls
                else func(h, y_n_array, y_np1_array, *rest[1:], **kwargs)
            )
        else:
            result_array = (
                func(self_or_cls, h, y_n_array, *rest, **kwargs)
                if self_or_cls
                else func(h, y_n_array, *rest, **kwargs)
            )

        # Convert result back to PositionDict
        return PositionDict.from_array(result_array, y_n.data)

    return wrapper  # type: ignore


def date_diff_in_seconds(dt2, dt1):
    timedelta = dt2 - dt1
    return timedelta.days * 24 * 3600 + timedelta.seconds


def dhms_from_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return days, hours, minutes, seconds


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
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

    return timeit_wrapper
