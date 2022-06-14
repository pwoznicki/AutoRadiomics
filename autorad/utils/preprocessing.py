import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import SimpleITK as sitk
import typer
from nipype.interfaces.dcm2nii import Dcm2niix

from autorad.config.type_definitions import PathLike

log = logging.getLogger(__name__)

dicom_app = typer.Typer()
nrrd_app = typer.Typer()


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
    os.makedirs(output_path, exist_ok=True)
    sitk.WriteImage(img, str(output_path))
