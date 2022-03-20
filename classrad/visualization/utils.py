import itertools
from typing import Callable, List, Sequence, Tuple, Union

import numpy as np


# taken from
# https://vincentblog.xyz/posts/medical-images-in-python-computed-tomography
def window(
    image: np.ndarray,
    window_center: float,
    window_width: float,
):
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    window_image = (window_image - img_min) / (img_max - img_min) * 255.0

    return window_image


# values taken from https://radiopaedia.org/articles/windowing-ct
def window_with_preset(image, body_part):
    if body_part == "soft tissues":
        return window(image, window_center=40, window_width=350)
    elif body_part == "bone":
        return window(image, window_center=400, window_width=1800)
    elif body_part == "lung":
        return window(image, window_center=-600, window_width=1500)
    elif body_part == "brain":
        return window(image, window_center=40, window_width=80)
    elif body_part == "liver":
        return window(image, window_center=150, window_width=30)
    else:
        raise ValueError(f"Unknown body part: {body_part}")


# taken from monai.transforms.utils
def is_positive(img):
    return img > 0


def crop_volume_from_coords(coords_start, coords_end, vol):
    return vol[
        coords_start[0] : coords_end[0],
        coords_start[1] : coords_end[1],
        coords_start[2] : coords_end[2],
    ]


# adapted from monai.transforms.utils
def generate_spatial_bounding_box(
    img: np.ndarray,
    select_fn: Callable = is_positive,
    margin: Union[Sequence[int], int] = 0,
) -> Tuple[List[int], List[int]]:
    data = img
    data = np.any(select_fn(data), axis=0)
    ndim = len(data.shape)
    if isinstance(margin, int):
        margin_by_dim = (margin,) * ndim
    else:
        margin_by_dim = margin
    for m in margin_by_dim:
        if m < 0:
            raise ValueError("Margin value should not be a negative number.")

    box_start = [0] * ndim
    box_end = [0] * ndim

    for di, ax in enumerate(
        itertools.combinations(reversed(range(ndim)), ndim - 1)
    ):
        dt = data.any(axis=ax)
        if not np.any(dt):
            # if no foreground, return all zero bounding box coords
            return [0] * ndim, [0] * ndim

        min_d = max(np.argmax(dt) - margin_by_dim[di], 0)  # type: ignore
        max_d = max(
            data.shape[di] - max(np.argmax(dt[::-1]) - margin_by_dim[di], 0),  # type: ignore
            min_d + 1,
        )
        box_start[di], box_end[di] = min_d, max_d

    return box_start, box_end
