import logging
from tqdm import tqdm
from typing import Callable, Sequence

import numpy as np

from autorad.utils import io

log = logging.getLogger(__name__)


def assert_dimensionality(arr, n_dims=3):
    assert (
        arr.ndim == n_dims
    ), f"Array has {arr.ndim} dimensions, expected {n_dims}"


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


def assert_no_empty_slice_in_3D_mask(array):
    assert (array.sum(axis=(1, 2)) > 0).all(), "Array has empty slices"


def assert_has_n_labels(array, expected_n_labels=1, exclude_zero=True):
    found_n_labels = np.unique(array).shape[0]
    if exclude_zero:
        found_n_labels -= 1
    assert (
        found_n_labels == expected_n_labels
    ), f"Array has {found_n_labels} labels, expected {expected_n_labels}"


def assert_is_binary(array):
    unique_values = np.unique(array)
    assert (
        unique_values.shape[0] == 2
    ), f"Array is not binary, found values: {unique_values}"
    if not np.array_equal(unique_values, [0, 1]):
        log.warning(f"Expected values [0 1], found values: {unique_values}")


def check_assertion_from_paths(assert_fn, paths: Sequence[str]):
    args = [io.load_array(path) for path in paths]
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
    assert_fn: Callable,
    paths: Sequence[str | tuple[str, str]],
    raise_error=True,
):
    log.info(f"Checking {assert_fn.__name__}...")
    asserts = [
        check_assertion_from_paths(assert_fn, pack_as_list(path))
        for path in tqdm(paths)
    ]
    failed_asserts = [a for a in asserts if a is not None]
    if len(failed_asserts) > 0:
        if raise_error:
            raise AssertionError(" ".join(failed_asserts))
        else:
            log.error(" ".join(failed_asserts))
