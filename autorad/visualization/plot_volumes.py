import functools
import itertools
import logging
import tempfile
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

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


def is_positive(img):
    return img > 0


def generate_spatial_bounding_box(
    img: np.ndarray,
    select_fn: Callable = is_positive,
    margin: Union[Sequence[int], int] = 0,
    allow_smaller: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Generate the spatial bounding box of foreground in the image with start-end positions (inclusive).
    Users can define arbitrary function to select expected foreground from the whole image or specified channels.
    And it can also add margin to every dim of the bounding box.
    The output format of the coordinates is:

        [1st_spatial_dim_start, 2nd_spatial_dim_start, ..., Nth_spatial_dim_start],
        [1st_spatial_dim_end, 2nd_spatial_dim_end, ..., Nth_spatial_dim_end]

    If `allow_smaller`, the bounding boxes edges are aligned with the input image edges.
    This function returns [0, 0, ...], [0, 0, ...] if there's no positive intensity.

    Args:
        img: a "channel-first" image of shape (C, spatial_dim1[, spatial_dim2, ...]) to generate bounding box from.
        select_fn: function to select expected foreground, default is to select values > 0.
        channel_indices: if defined, select foreground only on the specified channels
            of image. if None, select foreground on the whole image.
        margin: add margin value to spatial dims of the bounding box, if only 1 value provided, use it for all dims.
        allow_smaller: when computing box size with `margin`, whether allow the image size to be smaller
            than box size, default to `True`.
    """
    spatial_size = img.shape[1:]
    data = img
    data = select_fn(data).any(0)
    ndim = len(data.shape)
    for m in margin:
        if m < 0:
            raise ValueError("margin value should not be negative number.")

    box_start = [0] * ndim
    box_end = [0] * ndim

    for di, ax in enumerate(
        itertools.combinations(reversed(range(ndim)), ndim - 1)
    ):
        dt = data
        if len(ax) != 0:
            dt = np.any(dt, ax)

        if not dt.any():
            # if no foreground, return all zero bounding box coords
            return [0] * ndim, [0] * ndim

        arg_max = np.where(dt == dt.max())[0]
        min_d = arg_max[0] - margin[di]
        max_d = arg_max[-1] + margin[di] + 1
        if allow_smaller:
            min_d = max(min_d, 0)
            max_d = min(max_d, spatial_size[di])

        box_start[di] = min_d
        box_end[di] = max_d

    return box_start, box_end


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
            select_fn = is_positive
        (self.coords_start, self.coords_end,) = generate_spatial_bounding_box(
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


def get_plotly_fig(img):
    fig = px.imshow(img)
    fig.update_layout(width=800, height=800)
    plotly_utils.hide_labels(fig)
    return fig


def plot_roi_compare_two_masks(
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


def plot_roi(image_path, mask_path):
    vols = BaseVolumes.from_nifti(image_path, mask_path, window=None)
    image_2D, mask_2D = vols.get_slices()
    img = overlay_mask_contour(image_2D, mask_2D)
    fig = get_plotly_fig(img)

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
        image_path = Path(image_path)
        mask_path = Path(mask_path)

        if resample:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                resampled_mask_path = tmpdir / "resampled_mask.nii.gz"
                spatial.resample_to_img(
                    to_resample=mask_path,
                    reference=image_path,
                    output_path=resampled_mask_path,
                )
                mask = io.load_array(resampled_mask_path)
        else:
            mask = io.load_array(mask_path)
        image = io.load_array(image_path)

        return cls(image, mask, *args, **kwargs)

    def crop_and_slice(self, volume: np.ndarray):
        result = self.preprocessor.transform(volume)
        return result

    def get_slices(self):
        image_2D = self.crop_and_slice(self.image)
        mask_2D = self.crop_and_slice(self.mask)
        return image_2D.T, mask_2D.T

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
                resampled_nifti_path = (
                    dir_path_obj / f"resampled_{name}.nii.gz"
                )
                spatial.resample_to_img(
                    to_resample=nifti_path,
                    reference=image_path,
                    output_path=resampled_nifti_path,
                )
                feature_map[name] = io.load_array(resampled_nifti_path)
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
