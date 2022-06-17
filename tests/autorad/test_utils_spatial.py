import numpy as np
import SimpleITK as sitk
from conftest import prostate_data

from autorad.utils import spatial


def test_get_border_outside_mask_mm():
    seg_path = prostate_data["seg"]
    seg = sitk.ReadImage(str(seg_path))
    dilated = spatial.get_border_outside_mask_mm(seg, (10, 10, 0))
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
