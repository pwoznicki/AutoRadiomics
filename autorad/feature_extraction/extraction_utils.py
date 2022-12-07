from typing import Sequence


def filter_pyradiomics_names(names: Sequence[str]):
    """
    Filter features used in pyradiomics.
    """

    return [
        col
        for col in names
        if col.startswith(("original", "wavelet", "log-sigma"))
    ]
