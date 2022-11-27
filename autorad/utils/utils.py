import datetime
import logging
import os
import time

log = logging.getLogger(__name__)


def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        log.info(func.__name__ + " took " + str(end - start) + "sec")
        return result

    return wrapper


def calculate_age(dob):
    """
    Calculate the age of a person from his date of birth.
    """
    today = datetime.datetime.now()
    return (
        today.year
        - dob.year
        - ((today.month, today.day) < (dob.month, dob.day))
    )


def calculate_age_at(date, dob):
    """
    Calculate the age of a person from his date of birth.

    """
    return (
        date.year - dob.year - ((date.month, date.day) < (dob.month, dob.day))
    )


def calculate_time_between(date1, date2):
    """
    Calculate the time between two dates.
    """
    return (date2 - date1).days


def set_n_jobs(n_jobs):
    """
    Set the number of parallel processes used by pyradiomics.
    """
    if n_jobs == -1:
        return os.cpu_count()
    else:
        return n_jobs
