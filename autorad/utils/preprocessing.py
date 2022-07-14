import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import SimpleITK as sitk
import typer
from joblib import Parallel, delayed
from nipype.interfaces.dcm2niix import Dcm2niix

from autorad.config.type_definitions import PathLike
from autorad.data.dataset import ImageDataset
from autorad.utils import spatial

log = logging.getLogger(__name__)

dicom_app = typer.Typer()
nrrd_app = typer.Typer()


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
        Path(output_dir) / f"{id_}_border_mask.nii.gz" for id_ in dataset.ids
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
    ids, images, masks = [], [], []
    for id_ in os.listdir(data_dir):
        id_dir = os.path.join(data_dir, id_)
        if not os.path.isdir(id_dir):
            continue
        case_paths = Path(id_dir).glob("*.nii.gz")
        image_paths = [p for p in case_paths if image_stem in p.name]
        if not len(image_paths) == 1:
            log.error(
                f"Expected 1, found {len(image_paths)} images for ID={id_}"
            )
            continue
        image_path = image_paths[0]
        for fname in os.listdir(id_dir):
            if mask_stem in fname:
                mask_path = os.path.join(id_dir, fname)
                if relative:
                    image_path = os.path.relpath(image_path, data_dir)
                    mask_path = os.path.relpath(mask_path, data_dir)

                ids.append(id_)
                images.append(str(image_path))
                masks.append(str(mask_path))
    return pd.DataFrame(
        {"case_ID": ids, "image_path": images, "segmentation_path": masks}
    )


def get_paths_with_separate_folder_per_case(
    data_dir: PathLike,
    image_stem: str = "image",
    mask_stem: str = "segmentation",
    relative: bool = False,
) -> pd.DataFrame:
    ids, images, masks = [], [], []
    for id_ in os.listdir(data_dir):
        id_dir = os.path.join(data_dir, id_)
        if not os.path.isdir(id_dir):
            continue
        image_path = os.path.join(id_dir, f"{image_stem}.nii.gz")
        mask_path = os.path.join(id_dir, f"{mask_stem}.nii.gz")
        if not os.path.exists(image_path):
            log.error(f"Image for ID={id_} does not exist ({image_path})")
            continue
        if not os.path.exists(mask_path):
            log.error(f"Mask for ID={id_} does not exist ({mask_path})")
            continue
        if relative:
            image_path = os.path.relpath(image_path, data_dir)
            mask_path = os.path.relpath(mask_path, data_dir)

        ids.append(id_)
        images.append(image_path)
        masks.append(mask_path)
    return pd.DataFrame(
        {"ID": ids, "image_path": images, "segmentation_path": masks}
    )


def get_dcm2niix_converter(
    dicom_dir: Path, save_dir: Path, out_filename: Optional[str] = None
):
    """
    Args:
        dicom_dir: directory with DICOM Study
        save_dir: directory where to save the nifti
        out_filename: what to include in filenames (from dcm2niix)
    Returns:
        converter: nipype Dcm2niix object based on rodenlab dcm2niix with basic
                   default parameters to merge Dicom series into 3d Volumes
    """
    converter = Dcm2niix()
    converter.inputs.source_dir = str(dicom_dir)
    converter.inputs.output_dir = str(save_dir)
    converter.inputs.merge_imgs = True
    if out_filename:
        converter.inputs.out_filename = out_filename

    return converter


@dicom_app.command()
def dicom_to_nifti(
    input_dir: str,
    output_dir: str,
    subdir_name: str = "",
    out_filename: Optional[str] = None,
):
    """
    Args:
        input_dir: absolute path to the directory with all the cases containing
                dicoms
        output_dir: absolute path to the directory where to save nifties
        subdir_name: optional name of subdirectory within case dir
    """
    for id_ in os.listdir(input_dir):
        dicom_dir = Path(input_dir) / id_ / subdir_name
        save_dir = Path(output_dir) / id_ / subdir_name
        save_dir.mkdir(exist_ok=True, parents=True)
        if not dicom_dir.exists():
            raise FileNotFoundError(f"Dicom directory {dicom_dir} not found.")
        converter = get_dcm2niix_converter(dicom_dir, save_dir, out_filename)
        converter.run()


@nrrd_app.command()
def nrrd_to_nifti(nrrd_path: str, output_path: str):
    """
    Converts an image in NRRD format to NifTI format.
    """
    img = sitk.ReadImage(nrrd_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(img, str(output_path))
