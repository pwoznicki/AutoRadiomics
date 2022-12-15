import datetime
import logging
import os
import time

log = logging.getLogger(__name__)


def get_not_none_kwargs(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}


def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        log.info(func.__name__ + " took " + str(end - start) + "sec")
        return result

    return wrapper


def calculate_age(date_of_birth):
    """
    Calculate the age of a person from his date of birth.
    """
    today = datetime.datetime.now()
    return (
        today.year
        - date_of_birth.year
        - ((today.month, today.day) < (date_of_birth.month, date_of_birth.day))
    )


def calculate_age_at(date, date_of_birth):
    """
    Calculate the age of a person from his date of birth.

    """
    return (
        date.year
        - date_of_birth.year
        - ((date.month, date.day) < (date_of_birth.month, date_of_birth.day))
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
