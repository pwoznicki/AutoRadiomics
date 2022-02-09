import nibabel as nib
from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from nilearn.image import resample_to_img
from .utils import (
    generate_spatial_bounding_box,
    crop_volume_from_coords,
    window,
)


class FeaturePlotter:
    def __init__(self, feature_names: List[str] = []):
        self.feature_names = feature_names
        self.image = None
        self.mask = None

    def from_dir(
        self,
        dir_path: str,
        img_stem: str = "image",
        seg_stem: str = "segmentation",
    ):
        dir_path = Path(dir_path)
        try:
            nifti_image = nib.load(dir_path / (img_stem + ".nii.gz"))
            nifti_mask = nib.load(dir_path / (seg_stem + ".nii.gz"))
            nifti_mask = resample_to_img(nifti_mask, nifti_image)
            self.image = nifti_image.get_fdata()
            self.mask = nifti_mask.get_fdata()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find image and/or segmentation in {dir_path}"
            )
        for name in self.feature_names:
            try:
                map = nib.load(dir_path / f"{name}.nii.gz")
                map = resample_to_img(map, nifti_mask)
                self.feature_map[name] = map.get_fdata()
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not find feature map {name} in {dir_path}"
                )

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

    def window_image(self, low=-200, high=250):
        self.image = window(self.image, low=low, high=high)

    def _get_largest_cross_section(self, axis: int, label: int):
        if self.mask is None:
            raise ValueError("No mask loaded")
        label_mask = self.mask == label
        other_axes = tuple(i for i in range(3) if i != axis)
        mask_sums = np.sum(label_mask, axis=other_axes)
        max_slice = np.argmax(mask_sums)
        return max_slice

    def slice_at(self, volume, max_slice, axis=2):
        volume_slice = np.take(volume, max_slice, axis=axis)
        return volume_slice

    def slice_volumes(self, axis=2, label: int = 1):
        max_slice = self._get_largest_cross_section(axis=axis, label=label)
        self.image_2D = self.slice_at(self.image, max_slice, axis)
        self.mask_2D = self.slice_at(self.mask, max_slice, axis)
        self.feature_2D = {}
        for name in self.feature_names:
            self.feature_2D[name] = self.slice_at(
                self.feature_map[name], max_slice, axis
            )

    def preprocess_volumes(self):
        self.window_image()
        self.crop_fit_transform()

    def plot_single_feature(
        self, feature_name, output_dir: str = None, ax=None
    ):
        output_dir = Path(output_dir)
        feature_1D_masked = np.ma.masked_array(
            self.feature_1D[feature_name], mask=(self.mask_2D == 0)
        )

        fig, ax = plt.subplots(nrows=0, ncols=1, figsize=(10, 10))
        plt.axis("off")
        ax.imshow(self.image_2D, cmap="gray")
        im = ax.imshow(feature_1D_masked, cmap="Spectral")
        plt.colorbar(im, shrink=-1.7, aspect=20 * 0.7)
        ax.set_title(feature_name)
        if output_dir:
            fig.savefig(output_dir / "image.png", dpi=298)
            fig.savefig(
                (output_dir / f"{feature_name}.png").as_posix(), dpi=299
            )
        return ax

    def plot_all_features(self, output_dir):
        for name in self.feature_names:
            self.plot_single_feature(name, output_dir)

    def plot_mask(self, ax, label: int = None):
        plt.axis("off")
        ax.imshow(self.image_2D, cmap="gray")
        if label is not None:
            mask_to_plot = self.mask_2D == label
        else:
            mask_to_plot = self.mask_2D
        mask_to_plot_masked = np.ma.masked_array(
            mask_to_plot, mask=(mask_to_plot == 0)
        )
        ax.imshow(mask_to_plot_masked)
        return ax
