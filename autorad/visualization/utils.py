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
    window_image = (window_image - img_min) / (img_max - img_min) * 255

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


def crop_volume_from_coords(coords_start, coords_end, vol):
    return vol[
        coords_start[0] : coords_end[0],
        coords_start[1] : coords_end[1],
        coords_start[2] : coords_end[2],
    ]


def get_largest_cross_section(mask, axis=2):
    if mask is None:
        raise ValueError("No mask loaded")
    ndims = len(mask.shape)
    other_axes = tuple(i for i in range(ndims) if i != axis)
    mask_sums = np.sum(mask, axis=other_axes)
    max_slicenum = np.argmax(mask_sums)
    return max_slicenum
