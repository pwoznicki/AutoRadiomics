import itertools
import logging
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    ResampleToMatchd,
)

log = logging.getLogger(__name__)


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
    window_image_scaled = (window_image - img_min) / (img_max - img_min) * 255

    return window_image_scaled.astype(int)


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


def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        log.info(func.__name__ + " took " + str(end - start) + "sec")
        return result

    return wrapper


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
    img = sitk.ReadImage(str(img_path))
    isotropic_img = resample_to_isotropic_sitk(
        image=img,
        interpolation=interpolation,
    )
    sitk.WriteImage(isotropic_img, str(output_path))


def load_and_resample_to_match(
    to_resample, reference, interpolation="nearest"
):
    """
    Args:
        to_resample: Path to the image to resample.
        reference: Path to the reference image.
    """
    data_dict = {"img": to_resample, "ref": reference}
    transforms = Compose(
        [
            LoadImaged(("img", "ref")),
            EnsureChannelFirstd(("img", "ref")),
            ResampleToMatchd("img", "ref_meta_dict", mode=interpolation),
        ]
    )
    result = transforms(data_dict)

    return result["img"][0], result["ref"][0]


def resample_to_img_sitk(img, target_img, interpolation="nearest"):
    """Resample image to target image.
    Both images should be sitk.Image instances.
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
    img_path, ref_path, output_path=None, interpolation="nearest"
):
    """
    Wrapper for resample_to_img_sitk that takes in paths instead of
    sitk.Image.
    """
    if output_path is None:
        output_path = img_path
    nifti = sitk.ReadImage(str(img_path))
    ref_nifti = sitk.ReadImage(str(ref_path))
    nifti_resampled = resample_to_img_sitk(
        img=nifti, target_img=ref_nifti, interpolation=interpolation
    )
    sitk.WriteImage(nifti_resampled, output_path)


def combine_nifti_masks(mask1_path, mask2_path, output_path):
    """
    Args:
        mask1_path: abs path to the first nifti mask
        mask2_path: abs path to the second nifti mask
        output_path: abs path to saved concatenated mask
    """
    if not Path(mask1_path).exists():
        raise FileNotFoundError(f"Mask {mask1_path} not found.")
    if not Path(mask2_path).exists():
        raise FileNotFoundError(f"Mask {mask2_path} not found.")

    mask1 = nib.load(mask1_path)
    mask2 = nib.load(mask2_path)

    matrix1 = mask1.get_fdata()
    matrix2 = mask2.get_fdata()
    assert matrix1.shape == matrix2.shape

    new_matrix = np.zeros(matrix1.shape)
    new_matrix[matrix1 == 1] = 1
    new_matrix[matrix2 == 1] = 2
    new_matrix = new_matrix.astype(int)

    new_mask = nib.Nifti1Image(
        new_matrix, affine=mask1.affine, header=mask1.header
    )
    nib.save(new_mask, output_path)


def relabel_mask(mask_path: str, label_map: dict[int, int], save_path):
    """
    Relabel mask with a new label map.
    E.g. for for a prostate mask with two labels:
    1 for peripheral zone and 2 for transition zone,
    relabel_mask(mask_path, {1: 1, 2: 1}) would merge both zones
    into label 1.
    """
    if not Path(mask_path).exists():
        raise FileNotFoundError(f"Mask {mask_path} not found.")
    mask = nib.load(mask_path)
    matrix = mask.get_fdata()
    n_found_labels = len(np.unique(matrix))
    if n_found_labels != len(label_map) + 1:
        raise ValueError(
            f"Number of unique labels in the mask is {n_found_labels}\
              and label map has {len(label_map)} items."
        )
    new_matrix = np.zeros(matrix.shape)
    for old_label, new_label in label_map.items():
        new_matrix[matrix == old_label] = new_label
    new_mask = nib.Nifti1Image(
        new_matrix, affine=mask.affine, header=mask.header
    )
    nib.save(new_mask, save_path)


def separate_nifti_masks(
    combined_mask_path, output_dir, label_dict=None, overwrite=False
):
    """
    Split multilabel nifti mask into separate binary nifti files.
    Args:
        combined_mask_path: abs path to the combined nifti mask
        output_dir: abs path to the directory to save separated masks
        label_dict: (optional) dictionary with names for each label
    """
    if not Path(combined_mask_path).exists():
        raise FileNotFoundError(f"Mask {combined_mask_path} not found.")

    mask = nib.load(combined_mask_path)
    matrix = mask.get_fdata()

    labels = np.unique(matrix).astype(int)
    labels = labels[labels != 0]

    assert sorted(labels), sorted(list(label_dict.keys()))

    for label in labels:
        new_matrix = np.zeros(matrix.shape)
        new_matrix[matrix == label] = 1
        new_matrix = new_matrix.astype(int)

        new_mask = nib.Nifti1Image(
            new_matrix, affine=mask.affine, header=mask.header
        )

        if label_dict is not None:
            label_name = label_dict[label]
        else:
            label_name = label

        output_path = Path(output_dir) / f"seg_{label_name}.nii.gz"
        if (not output_path.exists()) or (overwrite is True):
            nib.save(new_mask, output_path)
