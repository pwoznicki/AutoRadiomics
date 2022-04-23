import os
from pathlib import Path
from typing import Optional

import SimpleITK as sitk
import typer
from nipype.interfaces.dcm2nii import Dcm2niix

dicom_app = typer.Typer()
nrrd_app = typer.Typer()


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
def nrrd_to_nifti(nrrd_path: str, output_dir: str):
    """
    Converts an image in NRRD format to NifTI format.
    """
    img = sitk.ReadImage(nrrd_path)
    output_path = Path(output_dir) / (Path(nrrd_path).stem + ".nii.gz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(output_path))
