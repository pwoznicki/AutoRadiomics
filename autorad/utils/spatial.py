from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Callable, Sequence

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from autorad.config.type_definitions import PathLike
from autorad.utils import io

log = logging.getLogger(__name__)


def get_border_outside_mask_mm(
    mask_path: PathLike,
    margin: float | Sequence[float],
    output_path: PathLike,
):
    """Wrapper that takes in paths
    instead of sitk.Image.
    """
    mask = sitk.ReadImage(str(mask_path))
    border_mask = get_border_outside_mask_mm_sitk(mask, margin=margin)
    sitk.WriteImage(border_mask, str(output_path))


def get_border_outside_mask_mm_sitk(mask, margin: float | Sequence[float]):
    dilated_mask = dilate_mask_mm_sitk(mask, margin=margin)
    # subtract mask from dilated mask to get border
    border = dilated_mask - mask
    return border


def dilate_mask_mm(
    mask_path: PathLike,
    margin: float | Sequence[float],
    output_path: PathLike,
):
    """Wrapper that takes in paths
    instead of sitk.Image.
    """
    mask = sitk.ReadImage(str(mask_path))
    dilated_mask = dilate_mask_mm_sitk(mask, margin=margin)
    sitk.WriteImage(dilated_mask, str(output_path))


def dilate_mask_mm_sitk(mask, margin: float | Sequence[float]):
    """Dilate a mask in 3D by a margin given in mm.
    Accepts margin as a single float number or sequence of 3 floats."""
    if isinstance(margin, int):
        margins = np.array([margin, margin, margin])
    else:
        margins = np.array(margin)
    if margins.size != 3:
        raise ValueError("margin must be a float or a 3-tuple")
    spacing = np.array(mask.GetSpacing())
    if any((m > 0) and (m < s) for m, s in zip(margins, spacing)):
        raise ValueError(
            f"Margin = {margin} mm cannot be non-zero and smaller than the spacing = {spacing}"
        )
    dilation_in_voxels = margins / spacing
    dilation_in_voxels = tuple(
        int(np.round(voxel)) for voxel in dilation_in_voxels
    )
    real_dilation_in_mm = dilation_in_voxels * np.array(spacing)
    log.info(f"Dilating by a margin in mm: {real_dilation_in_mm}")
    dilated_mask = sitk.BinaryDilate(
        mask,
        dilation_in_voxels,
    )
    return dilated_mask


def center_of_mass(array: np.ndarray) -> list[float]:
    total = array.sum()
    ndim = len(array.shape)
    result = []
    for dim in range(ndim):
        other_dims = tuple(i for i in range(ndim) if i != dim)
        coord = (
            array.sum(axis=(other_dims)) @ range(array.shape[dim])
        ) / total
        result.append(coord)
    return result


def generate_bbox_around_mask_center(mask, bbox_size):
    """Mask a bounding box with fixed side length
    around the center of the input amask.
    """
    mask_center = center_of_mass(mask)
    mask_center = np.array(mask_center).round().astype(int)
    result = np.zeros_like(mask)
    margin_left = bbox_size // 2
    margin_right = bbox_size - margin_left
    low = mask_center - margin_left
    high = mask_center + margin_right
    result[
        max(0, low[0]) : min(mask.shape[0], high[0]),
        max(0, low[1]) : min(mask.shape[1], high[1]),
        max(0, low[2]) : min(mask.shape[2], high[2]),
    ] = 1
    return result


# taken from
# https://vincentblog.xyz/posts/medical-images-in-python-computed-tomography
def get_window(
    image: np.ndarray,
    window_center: float,
    window_width: float,
):
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    window_image_scaled = (window_image - img_min) / (img_max - img_min) * 255

    return window_image_scaled.astype(int)


# values taken from https://radiopaedia.org/articles/windowing-ct
def window_with_preset(image, window):
    if window == "soft tissues":
        return get_window(image, window_center=50, window_width=400)
    elif window == "bone":
        return get_window(image, window_center=400, window_width=1800)
    elif window == "lung":
        return get_window(image, window_center=-600, window_width=1500)
    elif window == "brain":
        return get_window(image, window_center=40, window_width=80)
    elif window == "liver":
        return get_window(image, window_center=30, window_width=150)
    else:
        raise ValueError(f"Unknown window setting: {window}")


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


def get_sitk_interpolator(interpolation):
    interpolation_mapper = {
        "nearest": sitk.sitkNearestNeighbor,
        "linear": sitk.sitkLinear,
        "bspline": sitk.sitkBSpline,
        "gaussian": sitk.sitkGaussian,
    }
    try:
        sitk_interpolator = interpolation_mapper[interpolation]
    except ValueError:
        raise ValueError(f"Interpolation {interpolation} not supported.")
    return sitk_interpolator


def resample_to_isotropic_sitk(
    image,
    interpolation="nearest",
    spacing=None,
    default_value=0,
    standardize_axes=False,
):
    """Implementation adapted from:
    https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/
    blob/master/Python/05_Results_Visualization.ipynb
    """
    original_spacing = image.GetSpacing()
    # Image is already isotropic, just return a copy.
    if all(spc == original_spacing[0] for spc in original_spacing):
        return sitk.Image(image)
    # Make image isotropic via resampling.
    original_size = image.GetSize()
    if spacing is None:
        spacing = min(original_spacing)
    new_spacing = [spacing] * image.GetDimension()
    new_size = [
        int(round(osz * ospc / spacing))
        for osz, ospc in zip(original_size, original_spacing)
    ]
    interpolator = get_sitk_interpolator(interpolation)
    new_direction = image.GetDirection()
    new_origin = image.GetOrigin()
    # Only need to standardize axes if user requested and the original
    # axes were not standard.
    if standardize_axes and not np.array_equal(
        np.array(new_direction), np.identity(image.GetDimension()).ravel()
    ):
        new_direction = np.identity(image.GetDimension()).ravel()
        # Compute bounding box for the original, non standard axes image.
        boundary_points = []
        for boundary_index in list(
            itertools.product(*zip([0, 0, 0], image.GetSize()))
        ):
            boundary_points.append(
                image.TransformIndexToPhysicalPoint(boundary_index)
            )
        max_coords = np.max(boundary_points, axis=0)
        min_coords = np.min(boundary_points, axis=0)
        new_origin = min_coords
        new_size = (
            ((max_coords - min_coords) / spacing).round().astype(int)
        ).tolist()
    isotropic_img = sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        interpolator,
        new_origin,
        new_spacing,
        new_direction,
        default_value,
        image.GetPixelID(),
    )
    return isotropic_img


def resample_to_isotropic(img_path, output_path, interpolation="nearest"):
    """Wrapper for resample_to_isotropic_sitk that takes in paths
    instead of sitk.Image.
    """
    if output_path is None:
        output_path = img_path
    img = io.load_sitk(img_path)
    isotropic_img = resample_to_isotropic_sitk(
        image=img,
        interpolation=interpolation,
    )
    io.save_sitk(isotropic_img, output_path)


def resample_to_img_sitk(
    img: sitk.Image, target_img: sitk.Image, interpolation="nearest"
) -> sitk.Image:
    """Resample an image to the same orientation and voxel size as a target image.

    Args:
        img: The image to resample.
        target_img: The target image to match orientation and voxel size.
        interpolation: The interpolation method to use.

    Returns:
        The resampled image.

    Raises:
        ValueError: If the interpolation method is not valid.
    """
    interpolator = get_sitk_interpolator(interpolation)
    resampled_img = sitk.Resample(
        img,
        target_img,
        sitk.Transform(),
        interpolator,
        0,
        img.GetPixelID(),
    )
    return resampled_img


def resample_to_img(
    to_resample: Path,
    reference: Path,
    output_path=None,
    interpolation="nearest",
):
    """
    Wrapper for resample_to_img_sitk that takes in paths instead of
    sitk.Image.
    """
    if output_path is None:
        output_path = to_resample
    nifti = io.load_sitk(to_resample)
    ref_nifti = io.load_sitk(reference)
    nifti_resampled = resample_to_img_sitk(
        img=nifti, target_img=ref_nifti, interpolation=interpolation
    )
    io.save_sitk(nifti_resampled, output_path)


def combine_nifti_masks(
    *masks: nib.Nifti1Image, use_separate_labels=True
) -> nib.Nifti1Image:
    """
    Combine multiple Nifti1Image masks into a single mask.

    Args:
        *masks: The masks to combine. All masks must have the same shape.
        use_separate_labels: If True, each mask will be given a unique
            label. If False, all masks will be given the same label.
    Returns:
        A new Nifti1Image with combined masks.

    Raises:
        ValueError: If fewer than two masks are provided, or the masks have
            different shapes.
    """
    if len(masks) < 2:
        raise ValueError("At least two masks must be provided")

    arrays = [mask.get_fdata() for mask in masks]
    shapes = [mask.shape for mask in masks]
    if len(set(shapes)) != 1:
        raise ValueError(
            f"All masks must have the same shape and found shapes: {shapes}"
        )

    new_matrix = np.zeros(shapes[0], dtype=int)

    for i, array in enumerate(arrays):
        if use_separate_labels:
            new_label = i + 1
        else:
            new_label = 1
        new_matrix[array != 0] = new_label

    return nib.Nifti1Image(new_matrix, affine=masks[0].affine)


def simple_relabel_fn(
    matrix: np.ndarray,
    label_map: dict[int, int],
    set_rest_to_zero: bool = False,
    background_value: int = 0,
) -> np.ndarray:
    """
    Relabel mask with a new label map.
    E.g. for for a prostate mask with two labels:
    1 for peripheral zone and 2 for transition zone,
    relabel_mask(mask_path, {1: 1, 2: 1}) would merge both zones
    into label 1.

    Args:
        matrix: The matrix of the multilabel mask.
        background_value: The value to set non-labeled voxels to.
        label_map: A dictionary mapping old labels to new labels.

    Returns:
        The relabeled mask.
    """
    if set_rest_to_zero:
        new_matrix = np.full_like(matrix, background_value)
    else:
        new_matrix = np.copy(matrix)
    for old_label, new_label in label_map.items():
        new_matrix[matrix == old_label] = new_label
    return new_matrix


def relabel_mask(
    mask: nib.Nifti1Image,
    relabel_fn: Callable[[np.ndarray, int], np.ndarray] = simple_relabel_fn,
    background_value: int = 0,
):
    """
    Relabel mask using a custom relabeling function.
    Args:
        mask: The mask to be relabeled.
        relabel_fn: A function that takes as input a 2D or 3D array representing a mask, and an integer value,
            and returns a 2D or 3D array with the mask relabeled according to the function's logic.
        background_value: The value to use for filling in parts of the mask that are not relabeled.
    Returns:
        The relabeled mask.
    """
    matrix = mask.get_fdata()
    new_matrix = relabel_fn(matrix, background_value)
    return nib.Nifti1Image(
        new_matrix.astype(np.uint8), affine=mask.affine, header=mask.header
    )


def create_binary_mask(mask, label):
    """
    Create a binary mask from a multilabel mask for the given label.
    Args:
        mask: The multilabel mask.
        label: The label value for which to create a binary mask.
    Returns:
        The binary mask.
    """
    matrix = mask.get_fdata()
    new_matrix = np.zeros(matrix.shape)
    new_matrix[matrix == label] = 1
    return nib.Nifti1Image(new_matrix, affine=mask.affine, header=mask.header)


def split_multilabel_nifti_masks(
    combined_mask_path: Path,
    output_dir: Path,
    label_dict: dict[int, str] | None = None,
    overwrite: bool = False,
    ignore_background: bool = True,
):
    """
    Split multilabel nifti mask into separate binary nifti files.
    Args:
        combined_mask_path: abs path to the combined nifti mask
        output_dir: abs path to the directory to save separated masks
        label_dict: (optional) dictionary with names for each label. If not provided,
            default names of the form "label_N" will be used, where N is the label value.
        overwrite: (optional) whether to overwrite existing files in the output directory
            with the same names as the separated masks.
        ignore_background: (optional) whether to ignore the background label, default True.
    Returns:
        A list of paths to the saved separated masks.
    """
    mask = io.load_nibabel(combined_mask_path)

    labels = np.unique(mask.get_fdata()).astype(int)
    if ignore_background:
        labels = labels[labels != 0]

    #     assert sorted(labels), sorted(list(label_dict.keys()))

    if label_dict is None:
        label_dict = {label: f"label_{label}" for label in labels}
    else:
        # Filter out any extra labels in label_dict that are not present in the combined mask
        label_dict = {
            label: label_name
            for label, label_name in label_dict.items()
            if label in labels
        }

    # Delete existing files in the output directory with the same names as the separated masks
    if overwrite:
        for label, label_name in label_dict.items():
            output_path = Path(output_dir) / f"seg_{label_name}.nii.gz"
            if output_path.exists():
                output_path.unlink()

    saved_masks = []
    for label, label_name in label_dict.items():
        new_mask = create_binary_mask(mask, label)
        output_path = Path(output_dir) / f"seg_{label_name}.nii.gz"
        io.save_nibabel(new_mask, output_path)
        saved_masks.append(output_path)

    return saved_masks
