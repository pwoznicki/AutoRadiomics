import numpy as np
import pandas as pd
import datetime
import nibabel as nib
import os
import SimpleITK as sitk
from pathlib import Path
from nilearn.image import resample_img

def convert_nrrd_to_nifti(nrrd_path, output_path):
    img = sitk.ReadImage(nrrd_path)
    sitk.WriteImage(img, output_path)


def resample_nifti(nifti_path, output_path, res=1.):
    '''resamples given nifti to an isotropic resolution of res'''
    assert(os.path.exists(nifti_path))
    nifti = nib.load(nifti_path)
    nifti_resampled = resample_img(nifti, target_affine=np.eye(3)*res, interpolation='nearest')
    nib.save(nifti_resampled, output_path)


def combine_nifti_masks(mask1_path, mask2_path, output_path):
    '''
    Args:
        mask1_path: abs path to the first nifti mask
        mask2_path: abs path to the second nifti mask
        output_path: abs path to saved concatenated mask
    '''
    assert(os.path.exists(mask1_path))
    assert(os.path.exists(mask2_path))

    mask1 = nib.load(mask1_path)
    mask2 = nib.load(mask2_path)

    matrix1 = mask1.get_fdata()
    matrix2 = mask2.get_fdata()
    assert(matrix1.shape == matrix2.shape)

    new_matrix = np.zeros(matrix1.shape)
    new_matrix[matrix1 == 1] = 1
    new_matrix[matrix2 == 1] = 2
    new_matrix = new_matrix.astype(int)

    new_mask = nib.Nifti1Image(new_matrix, affine=mask1.affine,
                               header=mask1.header)
    nib.save(new_mask, output_path)

def separate_nifti_masks(combined_mask_path, output_dir):
    '''
    Split multilabel nifti mask into separate binary nifti files.
    Args:
        combined_mask_path: abs path to the combined nifti mask
        output_path: abs path where to save separate masks
    '''
    assert(Path(combined_mask_path).exists())

    mask = nib.load(combined_mask_path)
    matrix = mask.get_fdata()

    labels = np.unique(matrix)
    labels = labels[labels != 0]

    for label in labels:
        label = int(label)
        new_matrix = np.zeros(matrix.shape)
        new_matrix[matrix == label] = 1
        new_matrix = new_matrix.astype(int)

        new_mask = nib.Nifti1Image(new_matrix, affine=mask.affine,
                               header=mask.header)
        output_path = Path(output_dir) / f"seg_{label}.nii.gz"
        nib.save(new_mask, output_path)

def get_peak_from_histogram(bins, bin_edges):
    '''
    Returns location of histogram peak.
    Can be applied on the output from np.histogram.
    In case of multiple equal peaks, returns locaion of the first one.

    Args :
        bins: values of histogram
        bin_edges: argument values at bin edges (len(bins)+1)

    Returns:
        peak_location: location of histogram peak (as a mean of mean edges)
    '''
    assert(len(bins) > 0)
    assert(len(bin_edges) == len(bins) + 1)
    try:
        peak_bin = np.argmax(bins)
        print(peak_bin, 'here i am!')
        peak_location = (bin_edges[peak_bin + 1] - bin_edges[peak_bin]) / 2

        return peak_location
    except:
        raise ValueError(f"Error processing the bins.")

def calculate_age(dob):
    """
    Calculate the age of a person from his date of birth.
    """
    today = datetime.datetime.now()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))    

