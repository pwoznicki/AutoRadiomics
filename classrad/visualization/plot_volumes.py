import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import monai.transforms.utils as monai_utils
import nibabel as nib
import numpy as np
import skimage
from nilearn.image import resample_to_img

from classrad.config.type_definitions import PathLike

from . import utils

log = logging.getLogger(__name__)


class BasePlotter:
    pass


def center_of_mass(array):
    return np.argwhere(array == 1).sum(0) // (array == 1).sum()


def center_of_mass_3D(array: np.ndarray):
    total = array.sum()
    # alternatively with np.arange as well
    x_coord = (array.sum(axis=(1, 2)) @ range(array.shape[0])) / total
    y_coord = (array.sum(axis=(0, 2)) @ range(array.shape[1])) / total
    z_coord = (array.sum(axis=(0, 1)) @ range(array.shape[2])) / total
    return x_coord, y_coord, z_coord


def select_constant_bbox_around_mask(mask, bbox_size):
    mask_center = center_of_mass(mask)
    result = np.zeros_like(mask)
    margin = bbox_size // 2
    result[
        mask_center[0] - margin : mask_center[0] + margin,
        mask_center[1] - margin : mask_center[1] + margin,
        mask_center[2] - margin : mask_center[2] + margin,
    ] = 1
    return result


class Cropper:
    """Performs non-zero cropping"""

    def __init__(self):
        self.coords_start = None
        self.coords_end = None

    def fit(
        self, mask: np.ndarray, margin=20, constant_bbox=False, bbox_size=20
    ):
        expanded_mask = np.expand_dims(mask, axis=0)
        if constant_bbox:
            select_fn = functools.partial(
                select_constant_bbox_around_mask, bbox_size=bbox_size
            )
        else:
            select_fn = monai_utils.is_positive
        coords_start, coords_end = monai_utils.generate_spatial_bounding_box(
            img=expanded_mask,
            select_fn=select_fn,
            margin=[margin, margin, margin],
        )
        log.info(
            f"Cropping the image of size {mask.shape} to the region from \
            {coords_start} to {coords_end}"
        )
        self.coords_start = coords_start
        self.coords_end = coords_end
        return self

    def transform(self, volume: np.ndarray):
        return utils.crop_volume_from_coords(
            self.coords_start, self.coords_end, volume
        )


class Slicer:
    def __init__(self):
        self.slicenum = None

    def fit(self, mask: np.ndarray, axis=2):
        slicenum = utils.get_largest_cross_section(mask, axis=axis)
        self.slicenum = slicenum
        return self

    def transform(self, volume: np.ndarray):
        return volume[:, :, self.slicenum]


@dataclass
class BaseSlices:
    image_2D: np.ndarray
    mask_2D: np.ndarray


def overlay_mask_contour(
    image_2D: np.ndarray,
    mask_2D: np.ndarray,
    label: int = 1,
    color=(204, 0, 0),
):
    mask_to_plot = mask_2D == label
    result_image = skimage.segmentation.mark_boundaries(
        image_2D, mask_to_plot, mode="outer", color=color
    )
    return result_image


class BaseVolumes:
    """
    Loading and processing of image and mask volumes.
    """

    def __init__(self, image, mask, constant_bbox=False):
        self.image = utils.window_with_preset(image, body_part="soft tissues")
        self.mask = mask
        self.cropper = Cropper().fit(self.mask, constant_bbox=constant_bbox)
        mask_cropped = self.cropper.transform(self.mask)
        self.slicer = Slicer().fit(mask_cropped)

    @classmethod
    def from_nifti(
        cls,
        image_path: PathLike,
        mask_path: PathLike,
        resample=False,
        *args,
        **kwargs,
    ):
        image_nifti = nib.load(image_path)
        mask_nifti = nib.load(mask_path)
        if resample:
            mask_nifti = resample_to_img(
                mask_nifti, image_nifti, interpolation="nearest"
            )
        image = image_nifti.get_fdata()
        mask = mask_nifti.get_fdata()
        return cls(image, mask, *args, **kwargs)

    def crop_and_slice(self, volume: np.ndarray):
        cropped = self.cropper.transform(volume)
        sliced = self.slicer.transform(cropped)
        return sliced

    def get_slices(self):
        image_2D = self.crop_and_slice(self.image)
        mask_2D = self.crop_and_slice(self.mask)
        return image_2D, mask_2D


class FeaturePlotter:
    def __init__(self, image, mask, feature_map: dict):
        self.volumes = BaseVolumes(image, mask)
        self.feature_map = feature_map
        self.feature_names = list(feature_map.keys())

    @classmethod
    def from_dir(cls, dir_path: str, feature_names: List[str]):
        dir_path_obj = Path(dir_path)
        try:
            nifti_image = nib.load(dir_path_obj / "image.nii.gz")
            nifti_mask = nib.load(dir_path_obj / "segmentation.nii.gz")
            image = nifti_image.get_fdata()
            mask = nifti_mask.get_fdata()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find image and/or segmentation in {dir_path}"
            )
        feature_map = {}
        for name in feature_names:
            try:
                map = nib.load(dir_path_obj / f"{name}.nii.gz")
                map = resample_to_img(map, nifti_mask)
                feature_map[name] = map.get_fdata()
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find feature map {name} in {dir_path}"
                )
        return cls(image, mask, feature_map)

    def plot_single_feature(self, feature_name, output_dir: str, ax=None):
        image_2D = self.volumes.crop_and_slice(self.volumes.image)
        mask_2D = self.volumes.crop_and_slice(self.volumes.mask)
        feature_2D = self.volumes.crop_and_slice(
            self.feature_map[feature_name]
        )
        feature_2D_masked = np.ma.masked_array(feature_2D, mask=(mask_2D == 0))

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        plt.axis("off")
        ax.imshow(image_2D, cmap="gray")
        fig.savefig(Path(output_dir) / "image.png", dpi=300)
        im = ax.imshow(feature_2D_masked, cmap="Spectral")
        plt.colorbar(im, shrink=0.7, aspect=20 * 0.7)
        ax.set_title(feature_name)
        fig.savefig(
            (Path(output_dir) / f"{feature_name}.png").as_posix(), dpi=300
        )
        return ax

    def plot_all_features(self, output_dir):
        pass
        for name in self.feature_names:
            self.plot_single_feature(name, output_dir)
