import numpy as np
import nibabel as nib
import os


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


def get_peak_from_histogram(bins, bin_edges):
    '''
    Returns location of histogram peak.
    Can be applied on the output from np.histogram.

    Args :
        bins: values of histogram
        bin_edges: argument values at bin edges (len(bins)+1)

    Returns:
        peak_location: location of histogram peak (as a mean of mean edges)
    '''
    assert(len(bins) > 0)
    assert(len(bin_edges) == len(bins) + 1)
    peak_bin = np.argmax(bins)
    peak_location = (bin_edges[peak_bin + 1] - bin_edges[peak_bin]) / 2

    return peak_location
