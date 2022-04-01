import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
from numpy.typing import NDArray

from classrad.config.type_definitions import PathLike

from . import utils

log = logging.getLogger(__name__)


class BasePlotter:
    pass


class VolumeProcessor:
    pass


def get_slices(image, mask):
    slicenum = get_largest_cross_section(mask, axis=2)
    return BaseSlices(image[:, :, slicenum], mask[:, :, slicenum])


class BaseSlices:
    def __init__(self, image_2D: NDArray, mask_2D: NDArray):
        self.image_2D = image_2D
        self.mask_2D = mask_2D

    def plot(self, ax, label: Optional[int] = None):
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        plt.axis("off")
        ax.imshow(self.image_2D, cmap="gray")

        if label is not None:
            mask_to_plot = self.mask_2D == label
        else:
            mask_to_plot = self.mask_2D
        mask_to_plot_masked = np.ma.masked_array(
            mask_to_plot, mask=(self.mask_2D == 0)
        )
        ax.imshow(mask_to_plot_masked)
        return ax


class BaseVolumes:
    """
    Loading and processing of image and mask volumes.
    """

    def __init__(self, image: NDArray, mask: NDArray):
        self.image = image
        self.mask = mask
        self.nonzero_crop_fit()

    @classmethod
    def from_nifti(cls, image_path: PathLike, mask_path: PathLike):
        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        return cls(image, mask)

    def nonzero_crop_fit(self, margin=20):
        expanded_mask = np.expand_dims(self.mask, axis=0)
        coords_start, coords_end = utils.generate_spatial_bounding_box(
            img=expanded_mask, margin=[margin, margin, margin]
        )
        log.info(
            f"Cropping the image of size {self.mask.shape} to the region from \
            {coords_start} to {coords_end}"
        )
        self.coords_start = coords_start
        self.coords_end = coords_end
        return self

    def nonzero_crop_transform(self, volume):
        return utils.crop_volume_from_coords(
            self.coords_start, self.coords_end, volume
        )

    def nonzero_crop_transform_base_volumes(self):
        image_cropped = self.nonzero_crop_transform(self.image)
        mask_cropped = self.nonzero_crop_transform(self.mask)

        return image_cropped, mask_cropped


def slice_features(self):
    self.feature_2D = {}
    for name in self.feature_names:
        self.feature_2D[name] = self.get_slices(self.feature_map[name])


def process_volumes(volumes: BaseVolumes):
    image_cropped, mask_cropped = volumes.nonzero_crop_transform_base_volumes()
    slices = get_slices(image_cropped, mask_cropped)
    image_2D_windowed = utils.window_with_preset(
        slices.image_2D, body_part="soft tissues"
    )
    return image_2D_windowed


def get_largest_cross_section(mask, axis=2):
    if mask is None:
        raise ValueError("No mask loaded")
    ndims = len(mask.shape)
    other_axes = tuple(i for i in range(ndims) if i != axis)
    mask_sums = np.sum(mask, axis=other_axes)
    max_slicenum = np.argmax(mask_sums)
    return max_slicenum


class FeaturePlotter:
    def __init__(self, image, mask, feature_map: dict):
        self.base_volumes = BaseVolumes(image, mask)
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

    def crop_feature_maps(self):
        for name in self.feature_names:
            self.feature_map[name] = self.base_volumes.nonzero_crop_transform(
                self.feature_map[name]
            )

    def plot_single_feature(
        self, slice, feature_2D, feature_name, output_dir: str, ax=None
    ):
        feature_2D_masked = np.ma.masked_array(
            feature_2D[feature_name], mask=(slice.mask_2D == 0)
        )

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        plt.axis("off")
        ax.imshow(slice.image_2D, cmap="gray")
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
        # for name in self.feature_names:
        #     self.plot_single_feature(name, output_dir)
