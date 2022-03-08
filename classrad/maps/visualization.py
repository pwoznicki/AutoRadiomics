from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img

from .utils import (
    crop_volume_from_coords,
    generate_spatial_bounding_box,
    window,
)


class BaseVolumePlotter:
    def __init__(self, image, mask):
        self.image = image
        self.mask = mask


class FeaturePlotter(BaseVolumePlotter):
    def __init__(self, image, mask, feature_map: dict):
        super().__init__(image, mask)
        self.feature_map = feature_map
        self.feature_names = list(feature_map.keys())

    @classmethod
    def from_dir(cls, dir_path: str, feature_names: List[str]):
        dir_path = Path(dir_path)
        try:
            nifti_image = nib.load(dir_path / "image.nii.gz")
            nifti_mask = nib.load(dir_path / "segmentation.nii.gz")
            image = nifti_image.get_fdata()
            mask = nifti_mask.get_fdata()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find image and/or segmentation in {dir_path}"
            )
        feature_map = {}
        for name in feature_names:
            try:
                map = nib.load(dir_path / f"{name}.nii.gz")
                map = resample_to_img(map, nifti_mask)
                feature_map[name] = map.get_fdata()
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find feature map {name} in {dir_path}"
                )
        return cls(image, mask, feature_map)

    def _get_largest_cross_section(self, volume, axis=2):
        if self.mask is None:
            raise ValueError("No mask loaded")
        other_axes = tuple(i for i in range(3) if i != axis)
        mask_sums = np.sum(self.mask, axis=other_axes)
        max_slice = np.argmax(mask_sums)
        volume_slice = np.take(volume, max_slice, axis=axis)
        return volume_slice

    def get_slice(self, volume, axis=2):
        slicenum = self._get_largest_cross_section(axis)
        return volume[:, :, slicenum]

    def crop_fit_transform(self, margin=20):
        expanded_mask = np.expand_dims(self.mask, axis=0)
        coords_start, coords_end = generate_spatial_bounding_box(
            img=expanded_mask, margin=[margin, margin, margin]
        )
        print(
            f"Cropping the image of size {self.mask.shape} to the region from \
            {coords_start} to {coords_end}"
        )

        self.image = crop_volume_from_coords(
            coords_start, coords_end, self.image
        )
        self.mask = crop_volume_from_coords(
            coords_start, coords_end, self.mask
        )
        for name in self.feature_names:
            self.feature_map[name] = crop_volume_from_coords(
                coords_start, coords_end, self.feature_map[name]
            )

    def window_image(self):
        self.image = window(self.image, low=-200, high=250)

    def plot_single_feature(self, feature_name, output_dir: str, ax=None):
        output_dir = Path(output_dir)
        feature_2D_masked = np.ma.masked_array(
            self.feature_2D[feature_name], mask=(self.mask_2D == 0)
        )

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        plt.axis("off")
        ax.imshow(self.image_2D, cmap="gray")
        fig.savefig(output_dir / "image.png", dpi=300)
        im = ax.imshow(feature_2D_masked, cmap="Spectral")
        plt.colorbar(im, shrink=0.7, aspect=20 * 0.7)
        ax.set_title(feature_name)
        fig.savefig((output_dir / f"{feature_name}.png").as_posix(), dpi=300)
        return ax

    def slice_volumes(self):
        self.image_2D = self.get_slice(self.image)
        self.mask_2D = self.get_slice(self.mask)
        self.feature_2D = {}
        for name in self.feature_names:
            self.feature_2D[name] = self.get_slice(self.feature_map[name])

    def preprocess_volumes(self):
        self.window_image()
        self.crop_fit_transform()
        self.slice_volumes()

    def plot_all_features(self, output_dir):
        for name in self.feature_names:
            self.plot_single_feature(name, output_dir)

    def plot_mask(self, ax, label: int = None):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
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
