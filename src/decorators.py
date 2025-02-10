from datetime import datetime
from functools import wraps


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
