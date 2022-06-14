import functools
import logging
import warnings
from pathlib import Path
from typing import Optional

import monai.transforms.utils as monai_utils
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import skimage
from sklearn.pipeline import Pipeline

from autorad.config.type_definitions import PathLike
from autorad.utils import io, spatial
from autorad.visualization import plotly_utils

# suppress skimage
warnings.filterwarnings(action="ignore", category=UserWarning)

log = logging.getLogger(__name__)


class Cropper:
    """Performs non-zero cropping"""

    def __init__(self, bbox_size=50, margin=20):
        self.bbox_size = bbox_size
        self.margin = margin
        self.coords_start = None
        self.coords_end = None

    def fit(self, X: np.ndarray, y=None, constant_bbox=False):
        """X is a binary mask"""
        expanded_mask = np.expand_dims(X, axis=0)
        if constant_bbox:
            select_fn = functools.partial(
                spatial.generate_bbox_around_mask_center,
                bbox_size=self.bbox_size,
            )
        else:
            select_fn = monai_utils.is_positive
        (
            self.coords_start,
            self.coords_end,
        ) = monai_utils.generate_spatial_bounding_box(
            img=expanded_mask,
            select_fn=select_fn,
            margin=[self.margin, self.margin, self.margin],
        )
        return self

    def transform(self, volume: np.ndarray):
        log.info(
            f"Cropping the image of size {volume.shape} to the region from \
            {self.coords_start} to {self.coords_end}"
        )
        return spatial.crop_volume_from_coords(
            self.coords_start, self.coords_end, volume
        )


class Slicer:
    """Given a 3D volume, finds its largest cross-section"""

    def __init__(self, axis=2):
        self.slicenum = None
        self.axis = axis

    def fit(self, X: np.ndarray, y=None):
        """X is a binary mask"""
        slicenum = spatial.get_largest_cross_section(X, axis=self.axis)
        self.slicenum = slicenum
        return self

    def transform(self, volume: np.ndarray):
        indices = [slice(None)] * 3
        indices[self.axis] = self.slicenum

        return volume[tuple(indices)]


def normalize_roi(image_array, mask_array):
    image_values = image_array[mask_array > 0]
    roi_max = np.max(image_values)
    roi_min = np.min(image_values)
    image_clipped = np.clip(image_array, roi_min, roi_max)
    image_norm = (image_clipped - roi_min) / (roi_max - roi_min)
    return image_norm


def overlay_mask_contour(
    image_2D: np.ndarray,
    mask_2D: np.ndarray,
    label: int = 1,
    color=(1, 0, 0),  # red
    normalize=False,
):
    mask_to_plot = mask_2D == label
    if normalize:
        image_2D = normalize_roi(image_2D, mask_to_plot)
    image_to_plot = skimage.img_as_ubyte(image_2D)
    result_image = skimage.segmentation.mark_boundaries(
        image_to_plot, mask_to_plot, mode="outer", color=color
    )
    return result_image


def plot_compare_two_masks(
    image_path, manual_mask_path, auto_mask_path, axis=2
):
    manual_vols = BaseVolumes.from_nifti(
        image_path,
        manual_mask_path,
        constant_bbox=True,
        resample=True,
        axis=axis,
    )
    auto_vols = BaseVolumes.from_nifti(
        image_path,
        auto_mask_path,
        constant_bbox=True,
        resample=True,
        axis=axis,
    )
    image_2D, manual_mask_2D = manual_vols.get_slices()
    auto_mask_2D = manual_vols.crop_and_slice(auto_vols.mask)
    img_one_contour = overlay_mask_contour(
        image_2D,
        manual_mask_2D,
    )
    img_two_contours = overlay_mask_contour(
        img_one_contour,
        auto_mask_2D,
        color=(0, 0, 1),  # blue
    )
    fig = px.imshow(img_two_contours)
    fig.update_layout(width=800, height=800)
    plotly_utils.hide_labels(fig)

    return fig


class BaseVolumes:
    """Loading and processing of image and mask volumes."""

    def __init__(
        self,
        image,
        mask,
        label=1,
        constant_bbox=False,
        window="soft tissues",
        axis=2,
    ):
        self.image_raw = image
        if window is None:
            self.image = skimage.exposure.rescale_intensity(image)
        else:
            self.image = spatial.window_with_preset(
                self.image_raw, window=window
            )
        self.mask = mask == label
        self.axis = axis
        self.preprocessor = self.init_and_fit_preprocessor(constant_bbox)

    def init_and_fit_preprocessor(self, constant_bbox=False):
        preprocessor = Pipeline(
            [
                ("cropper", Cropper()),
                ("slicer", Slicer(axis=self.axis)),
            ]
        )
        preprocessor.fit(
            self.mask,
            cropper__constant_bbox=constant_bbox,
        )
        return preprocessor

    @classmethod
    def from_nifti(
        cls,
        image_path: PathLike,
        mask_path: PathLike,
        resample=False,
        *args,
        **kwargs,
    ):

        if resample:
            mask, image = spatial.load_and_resample_to_match(
                to_resample=mask_path,
                reference=image_path,
            )
        else:
            image = io.load_image(image_path)
            mask = io.load_image(mask_path)

        return cls(image, mask, *args, **kwargs)

    def crop_and_slice(self, volume: np.ndarray):
        result = self.preprocessor.transform(volume)
        return result

    def get_slices(self):
        image_2D = self.crop_and_slice(self.image)
        mask_2D = self.crop_and_slice(self.mask)
        return image_2D, mask_2D

    def plot_image(self):
        image_2D, _ = self.get_slices()
        fig = px.imshow(image_2D, color_continuous_scale="gray")
        plotly_utils.hide_labels(fig)
        return fig


class FeaturePlotter:
    """Plotting of voxel-based radiomics features."""

    def __init__(self, image_path, mask_path, feature_map: dict):
        self.volumes = BaseVolumes.from_nifti(
            image_path, mask_path, constant_bbox=True
        )
        self.feature_map = feature_map
        self.feature_names = list(feature_map.keys())

    @classmethod
    def from_dir(cls, dir_path: str, feature_names: list[str]):
        dir_path_obj = Path(dir_path)
        image_path = dir_path_obj / "image.nii.gz"
        mask_path = dir_path_obj / "segmentation.nii.gz"
        feature_map = {}
        for name in feature_names:
            nifti_path = dir_path_obj / f"{name}.nii.gz"
            try:
                feature_map[name], _ = spatial.load_and_resample_to_match(
                    nifti_path, image_path, interpolation="bilinear"
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Could not load feature map {name} in {dir_path}"
                )
        return cls(image_path, mask_path, feature_map)

    def plot_single_feature(
        self,
        feature_name: str,
        feature_range: Optional[tuple[float, float]] = None,
    ):
        image_2D, mask_2D = self.volumes.get_slices()
        feature_2D = self.volumes.crop_and_slice(
            self.feature_map[feature_name]
        )
        feature_2D[mask_2D == 0] = np.nan
        fig = px.imshow(image_2D, color_continuous_scale="gray")
        plotly_utils.hide_labels(fig)
        # Plot mask on top of fig, without margins
        heatmap_options = {
            "z": feature_2D,
            "showscale": False,
            "colorscale": "Spectral",
        }
        if feature_range:
            heatmap_options["zmin"] = feature_range[0]
            heatmap_options["zmax"] = feature_range[1]
        fig.add_trace(go.Heatmap(**heatmap_options))
        return fig

    def plot_all_features(self, output_dir, param_ranges):
        fig = self.volumes.plot_image()
        fig.write_image(Path(output_dir) / "image.png")
        for name, param_range in zip(self.feature_names, param_ranges):
            fig = self.plot_single_feature(name, param_range)
            fig.write_image(Path(output_dir) / f"{name}.png")
