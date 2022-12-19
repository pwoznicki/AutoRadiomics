import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from conftest import prostate_data

from autorad.utils import spatial


def test_get_border_outside_mask_mm():
    seg_path = prostate_data["seg"]
    seg = sitk.ReadImage(str(seg_path))
    dilated = spatial.get_border_outside_mask_mm_sitk(seg, (10, 10, 0))
    seg_arr = sitk.GetArrayFromImage(seg)
    dilated_arr = sitk.GetArrayFromImage(dilated)
    assert np.sum(seg_arr * dilated_arr) == 0


def test_dilate_mask_by_10mm():
    seg_path = prostate_data["seg"]
    seg = sitk.ReadImage(str(seg_path))
    dilated = spatial.dilate_mask_mm_sitk(seg, 10)
    ref_dilated_path = prostate_data["seg_dilated_10mm"]
    ref_dilated = sitk.ReadImage(str(ref_dilated_path))
    dilated_arr = sitk.GetArrayFromImage(dilated)
    ref_dilated_arr = sitk.GetArrayFromImage(ref_dilated)
    diff = np.sum(dilated_arr - ref_dilated_arr)
    assert abs(diff) / ref_dilated_arr.size < 0.05


def test_center_of_mass():
    arr = np.zeros((10, 10, 10))
    arr[0, 0, 0] = 1
    center = spatial.center_of_mass(arr)
    assert center == [0, 0, 0]

    arr2 = np.ones((10, 10, 10))
    center2 = spatial.center_of_mass(arr2)
    assert center2 == [4.5, 4.5, 4.5]  # with 0-indexing


def test_combine_nifti_masks():
    mask1_data = np.zeros((3, 3), dtype=int)
    mask1_data[2, 2] = 1
    mask1 = nib.Nifti1Image(mask1_data, np.eye(4))

    mask2_data = np.zeros((3, 3), dtype=int)
    mask2_data[1, 2] = 1
    mask2 = nib.Nifti1Image(mask2_data, np.eye(4))

    mask3_data = np.zeros((3, 3), dtype=int)
    mask3_data[0, 2] = 1
    mask3 = nib.Nifti1Image(mask3_data, np.eye(4))

    # separate labels
    combined_mask = spatial.combine_nifti_masks(mask1, mask2, mask3)
    expected_mask_data = np.zeros((3, 3), dtype=int)
    expected_mask_data[0, 2] = 3
    expected_mask_data[1, 2] = 2
    expected_mask_data[2, 2] = 1
    expected_mask = nib.Nifti1Image(expected_mask_data, np.eye(4))

    # all in one label
    combined_mask = spatial.combine_nifti_masks(
        mask1, mask2, mask3, use_separate_labels=False
    )
    expected_mask_data = np.zeros((3, 3), dtype=int)
    expected_mask_data[0, 2] = 1
    expected_mask_data[1, 2] = 1
    expected_mask_data[2, 2] = 1
    expected_mask = nib.Nifti1Image(expected_mask_data, np.eye(4))

    assert combined_mask.get_fdata().all() == expected_mask.get_fdata().all()


def test_simple_relabel_fn():
    matrix = np.array([[0, 1, 2], [3, 4, 5]])
    label_map = {1: 10, 2: 20}

    # keep the other classes as is
    expected = np.array([[0, 10, 20], [3, 4, 5]])
    result = spatial.simple_relabel_fn(matrix, label_map)
    assert (result == expected).all()

    # set the other classes to background
    expected = np.array([[0, 10, 20], [0, 0, 0]])
    result = spatial.simple_relabel_fn(
        matrix, label_map, set_rest_to_zero=True
    )
    assert (result == expected).all()


def test_generate_bbox_around_mask_center():
    arr = np.ones((11, 11, 11))
    arr[5, 5, 5] = 2
    bbox_mask = spatial.generate_bbox_around_mask_center(arr, bbox_size=5)
    assert bbox_mask.sum() == 5 * 5 * 5
    assert (bbox_mask[3:7, 3:7, 3:7] == 1).all()


def test_get_window():
    image = np.array(range(100))
    window_center = 50
    window_width = 20
    windowed_image = spatial.get_window(image, window_center, window_width)
    assert (windowed_image[:40] == 0).all()
    assert (windowed_image[60:] == 255).all()
    assert windowed_image[50] == 127


def test_get_largest_cross_section():
    mask = np.zeros((10, 10, 10))
    mask[0, 0, :] = 1
    mask[1, 1, 1] = 1
    max_across_0 = spatial.get_largest_cross_section(mask, axis=0)
    assert max_across_0 == 0
    max_across_1 = spatial.get_largest_cross_section(mask, axis=1)
    assert max_across_1 == 0
    max_across_2 = spatial.get_largest_cross_section(mask, axis=2)
    assert max_across_2 == 1


def test_create_binary_mask():
    # Create a multilabel mask with 3 labels
    matrix = np.array([[1, 2, 3], [3, 1, 2], [2, 3, 1]])
    mask = nib.Nifti1Image(matrix, affine=np.eye(4))
    binary_mask = spatial.create_binary_mask(mask, 2)
    assert binary_mask.shape == matrix.shape
    expected_binary_mask = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    assert np.isclose(binary_mask.get_fdata(), expected_binary_mask).all()


def test_split_multilabel_nifti_masks():
    # Create test input data
    input_data = np.array([[[1, 1, 2], [1, 2, 3], [0, 3, 0]]])
    input_mask = nib.Nifti1Image(input_data, np.eye(4))

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        input_path = temp_dir / "test_mask.nii.gz"
        nib.save(input_mask, input_path)

        # Create expected output data
        expected_data_1 = np.array([[[1, 1, 0], [1, 0, 0], [0, 0, 0]]])
        expected_mask_1 = nib.Nifti1Image(expected_data_1, np.eye(4))
        expected_path_1 = temp_dir / "seg_label_1.nii.gz"
        nib.save(expected_mask_1, expected_path_1)

        expected_data_2 = np.array([[[0, 0, 1], [0, 1, 0], [0, 0, 0]]])
        expected_mask_2 = nib.Nifti1Image(expected_data_2, np.eye(4))
        expected_path_2 = temp_dir / "seg_label_2.nii.gz"
        nib.save(expected_mask_2, expected_path_2)

        expected_data_3 = np.array([[[0, 0, 0], [0, 0, 1], [0, 1, 0]]])
        expected_mask_3 = nib.Nifti1Image(expected_data_3, np.eye(4))
        expected_path_3 = temp_dir / "seg_label_3.nii.gz"
        nib.save(expected_mask_3, expected_path_3)

        # Test with default label names
        output_paths = spatial.split_multilabel_nifti_masks(
            input_path, temp_dir
        )
        assert len(output_paths) == 3
        assert expected_path_1 in output_paths
        assert expected_path_2 in output_paths
        assert expected_path_3 in output_paths
        assert (
            nib.load(expected_path_1).get_fdata().all()
            == expected_data_1.all()
        )
        assert (
            nib.load(expected_path_2).get_fdata().all()
            == expected_data_2.all()
        )
        assert (
            nib.load(expected_path_3).get_fdata().all()
            == expected_data_3.all()
        )

        # Test with custom label names
        label_dict = {1: "one", 2: "two"}
        output_paths = spatial.split_multilabel_nifti_masks(
            input_path, temp_dir, label_dict=label_dict
        )
        assert len(output_paths) == 2
        expected_path_one = temp_dir / "seg_one.nii.gz"
        expected_path_two = temp_dir / "seg_two.nii.gz"
        assert expected_path_one in output_paths
        assert expected_path_two in output_paths

        assert (
            nib.load(expected_path_one).get_fdata().all()
            == expected_data_1.all()
        )
        assert (
            nib.load(expected_path_two).get_fdata().all()
            == expected_data_2.all()
        )
