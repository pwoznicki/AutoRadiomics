import datetime
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from nilearn.image import resample_img

log = logging.getLogger(__name__)


def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        log.info(func.__name__ + " took " + str(end - start) + "sec")
        return result

    return wrapper


def resample_nifti(nifti_path, output_path, res=1.0):
    """
    Resamples a nifti to an isotropic resolution of res
    Args:
        nifti_path: path to a NifTI image
        output_path: path where to save the resampled NifTI image
    """
    assert Path(nifti_path).exists()
    nifti = nib.load(nifti_path)
    nifti_resampled = resample_img(
        nifti, target_affine=np.eye(3) * res, interpolation="nearest"
    )
    nib.save(nifti_resampled, output_path)


def resample_to_img(img, target_img, interpolation="nearest"):
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
    resampled_img = sitk.Resample(
        img,
        target_img,
        sitk.Transform(),
        sitk_interpolator,
        0,
        img.GetPixelID(),
    )
    return resampled_img


def resample_to_nifti(
    nifti_path, ref_path, output_path=None, interpolation="nearest"
):
    """
    Resamples nifti to reference nifti, using nilearn.image.resample_to_img
    if output_path not give, overwrites the nifti_path.
    """
    nifti_path = str(nifti_path)
    ref_path = str(ref_path)
    if output_path is None:
        output_path = nifti_path
    nifti = sitk.ReadImage(nifti_path)
    ref_nifti = sitk.ReadImage(ref_path)
    nifti_resampled = resample_to_img(
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


def relabel_mask(mask_path: str, label_map: Dict[int, int], save_path):
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


def calculate_age_at(dob, date):
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


def get_pyradiomics_names(names: List[str]):
    """
    Filter features used in pyradiomics.
    """

    return [
        col
        for col in names
        if col.startswith(("original", "wavelet", "log-sigma"))
    ]


def set_n_jobs(n_jobs):
    """
    Set the number of parallel processes used by pyradiomics.
    """
    if n_jobs == -1:
        return os.cpu_count()
    else:
        return n_jobs
