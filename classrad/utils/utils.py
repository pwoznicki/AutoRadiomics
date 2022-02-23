import datetime
import time
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from nilearn.image import resample_img, resample_to_img


def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__ + " took " + str(end - start) + "sec")
        return result

    return wrapper


def convert_nrrd_to_nifti(nrrd_path, output_path):
    """
    Converts an image in NRRD format to NifTI format.
    """
    img = sitk.ReadImage(nrrd_path)
    sitk.WriteImage(img, output_path)


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


def resample_to_nifti(
    nifti_path, ref_path, output_path=None, interpolation="nearest"
):
    """
    Resamples nifti to reference nifti, using nilearn resample_to_img, with
    paths as arguments.
    """
    if output_path is None:
        output_path = nifti_path
    nifti = nib.load(nifti_path)
    ref_nifti = nib.load(ref_path)
    nifti_resampled = resample_to_img(nifti, ref_nifti, interpolation)
    print(
        f"Mask size resamopled from {nifti.get_fdata().shape} to \
            {nifti_resampled.get_fdata().shape} [mask_path={output_path}]"
    )
    nib.save(nifti_resampled, output_path)


def combine_nifti_masks(mask1_path, mask2_path, output_path):
    """
    Args:
        mask1_path: abs path to the first nifti mask
        mask2_path: abs path to the second nifti mask
        output_path: abs path to saved concatenated mask
    """
    assert Path(mask1_path).exists
    assert Path(mask2_path).exists

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


def separate_nifti_masks(
    combined_mask_path, output_dir, label_dict=None, overwrite=False
):
    """
    Split multilabel nifti mask into separate binary nifti files.
    Args:
        combined_mask_path: abs path to the combined nifti mask
        output_path: abs path where to save separate masks
        label_dict: (optional) dictionary with names for each label
    """
    assert Path(combined_mask_path).exists()

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


def get_peak_from_histogram(bins, bin_edges):
    """
    Returns location of histogram peak.
    Can be applied on the output from np.histogram.
    In case of multiple equal peaks, returns locaion of the first one.

    Args :
        bins: values of histogram
        bin_edges: argument values at bin edges (len(bins)+1)

    Returns:
        peak_location: location of histogram peak (as a mean of mean edges)
    """
    assert len(bins) > 0
    assert len(bin_edges) == len(bins) + 1
    try:
        peak_bin = np.argmax(bins)
        print(peak_bin, "here i am!")
        peak_location = (bin_edges[peak_bin + 1] - bin_edges[peak_bin]) / 2

        return peak_location
    except Exception:
        raise ValueError("Error processing the bins.")


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


def filter_pyradiomics_names(names: List[str]):
    """
    Filter features used in pyradiomics.
    """

    return [
        col for col in names if col.startswith(("original", "wavelet", "log"))
    ]
