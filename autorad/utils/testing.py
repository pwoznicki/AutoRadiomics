import logging
from typing import Callable, Sequence

import numpy as np

from autorad.utils import io

log = logging.getLogger(__name__)


def assert_equal_shape(arr1, arr2):
    assert (
        arr1.shape == arr2.shape
    ), f"Shape of arrays {arr1.shape} and {arr2.shape} are not equal"


def assert_has_nonzero(arr):
    assert (arr == 1).sum() > 0, "Array has only zeros"


def assert_has_nonzero_within_roi(arr, mask):
    assert (
        arr[mask == 1]
    ).sum() > 0, "Array has only zeros within the region where mask==1"


def assert_not_equal(arr1, arr2):
    assert not np.array_equal(arr1, arr2), "Arrays are the same"


def check_assertion_from_paths(assert_fn, paths: Sequence[str]):
    args = [io.load_image(path) for path in paths]
    try:
        assert_fn(*args)
        return None
    except AssertionError as e:
        msg = f"{e}, file: {paths} \n "
        return msg


def pack_as_list(arg):
    if isinstance(arg, str):
        result = [arg]
    else:
        result = list(arg)
    return result


def check_assertion_dataset(
    assert_fn: Callable, paths: Sequence[str | tuple[str, str]]
):

    asserts = [
        check_assertion_from_paths(assert_fn, pack_as_list(path))
        for path in paths
    ]
    failed_asserts = [a for a in asserts if a is not None]
    if len(failed_asserts) > 0:
        raise AssertionError(" ".join(failed_asserts))
