import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from joblib import Parallel, delayed

from autorad.config.type_definitions import PathLike
from autorad.data.dataset import ImageDataset
from autorad.utils import spatial

log = logging.getLogger(__name__)


def generate_border_masks(
    dataset: ImageDataset,
    margin_in_mm: float | Sequence[float],
    output_dir: PathLike,
    n_jobs: int = -1,
):
    """
    Generate a border mask (= mask with given margin around the original ROI)
    for each mask in the dataset.
    Returns a DataFrame extending ImageDataset.df with the additional column
    "dilated_mask_path_<margin_in_mm>".
    """
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    output_paths = [
        os.path.join(output_dir, f"{id_}_border_mask.nii.gz")
        for id_ in dataset.ids
    ]
    if n_jobs > 1:
        with Parallel(n_jobs) as parallel:
            parallel(
                delayed(spatial.get_border_outside_mask_mm)(
                    mask_path=mask_path,
                    margin=margin_in_mm,
                    output_path=output_path,
                )
                for mask_path, output_path in zip(
                    dataset.mask_paths, output_paths
                )
            )
    else:
        for mask_path, output_path in zip(dataset.mask_paths, output_paths):
            spatial.get_border_outside_mask_mm(
                mask_path,
                margin_in_mm,
                output_path,
            )
    result_df = dataset.df.copy()
    result_df[f"border_mask_path_{margin_in_mm}mm"] = output_paths

    return result_df


def get_paths_with_separate_folder_per_case_loose(
    data_dir: PathLike,
    image_stem: str = "image",
    mask_stem: str = "segmentation",
    relative: bool = False,
) -> pd.DataFrame:
    ids, image_paths, mask_paths = [], [], []
    for id_ in os.listdir(data_dir):
        id_dir = os.path.join(data_dir, id_)
        if not os.path.isdir(id_dir):
            continue
        case_image_paths = Path(id_dir).glob("*.nii.gz")
        case_image_paths = [
            p for p in case_image_paths if image_stem in p.name
        ]
        if not len(case_image_paths) == 1:
            log.error(
                f"Expected 1, found {len(image_paths)} images for ID={id_}"
            )
            continue
        image_path = case_image_paths[0]
        for fname in os.listdir(id_dir):
            if mask_stem in fname:
                mask_path = os.path.join(id_dir, fname)

                ids.append(id_)
                image_paths.append(str(image_path))
                mask_paths.append(str(mask_path))
    if relative:
        image_paths = make_relative(image_paths, data_dir)
        mask_paths = make_relative(mask_paths, data_dir)
    path_df = paths_to_df(ids, image_paths, mask_paths)

    return path_df


def get_paths_with_separate_folder_per_case(
    data_dir: PathLike,
    image_stem: str = "image",
    mask_stem: str = "segmentation",
    relative: bool = False,
) -> pd.DataFrame:
    ids, image_paths, mask_paths = [], [], []
    for id_ in os.listdir(data_dir):
        id_dir = os.path.join(data_dir, id_)
        if not os.path.isdir(id_dir):
            continue
        image_path = os.path.join(id_dir, f"{image_stem}.nii.gz")
        mask_path = os.path.join(id_dir, f"{mask_stem}.nii.gz")
        if not os.path.exists(image_path):
            log.warning(f"Image for ID={id_} does not exist ({image_path})")
            continue
        if not os.path.exists(mask_path):
            log.warning(f"Mask for ID={id_} does not exist ({mask_path})")
            continue
        ids.append(id_)
        image_paths.append(image_path)
        mask_paths.append(mask_path)
    if relative:
        image_paths = make_relative(image_paths, data_dir)
        mask_paths = make_relative(mask_paths, data_dir)
    path_df = paths_to_df(ids, image_paths, mask_paths)

    return path_df


def get_paths_with_separate_image_seg_folders(
    image_dir: PathLike,
    mask_dir: PathLike,
    relative_to: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Get paths of images and segmentations when all images are in one folder
    and all segmentations are in another folder. It assumes the file names
    for images and segmentation are the same, or image filenames may have optional
    suffix _XXXX, stemming from nnUNet.

    Args:
        image_dir (PathLike): Path to the folder with the images.
        mask_dir (PathLike): Path to the folder with the segmentations.
        relative_to (Optional[PathLike], optional): put a root directory to get relative paths.
            If None, return absolute paths.

    Returns:
        tuple[List[str], List[str], List[str]]: IDs, image paths, and mask paths
    """
    images = list(Path(image_dir).glob("*.nii.gz"))
    masks = list(Path(mask_dir).glob("*.nii.gz"))
    mask_ids = [p.stem.split(".")[0] for p in masks]
    mask_dict = {id_: str(p) for id_, p in zip(mask_ids, masks)}
    ids_matched, images_matched, masks_matched = [], [], []
    for image in images:
        id_ = image.stem.split(".")[0]
        if id_.endswith("_0000"):
            id_ = id_[:-5]
        if id_ not in mask_ids:
            log.error(f"No mask found for ID={id_}")
        else:
            ids_matched.append(id_)
            images_matched.append(str(image))
            masks_matched.append(str(mask_dict[id_]))
    if relative_to is not None:
        images_matched = make_relative(images_matched, relative_to)
        masks_matched = make_relative(masks_matched, relative_to)

    path_df = paths_to_df(ids_matched, images_matched, masks_matched)

    return path_df


def paths_to_df(ids, image_paths, mask_paths):
    df = pd.DataFrame(
        {"ID": ids, "image_path": image_paths, "segmentation_path": mask_paths}
    )
    return df


def make_relative(paths: List[PathLike], root_dir):
    return [os.path.relpath(p, root_dir) for p in paths]
