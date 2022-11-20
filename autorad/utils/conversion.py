import logging
from pathlib import Path
from typing import Optional

import SimpleITK as sitk
import typer
from monai.transforms import Compose, LoadImage, SaveImage
from nipype.interfaces.dcm2nii import Dcm2niix

from autorad.config.type_definitions import PathLike

log = logging.getLogger(__name__)

dicom_app = typer.Typer()
nrrd_app = typer.Typer()


def convert_dataset_to_nifti(data_dir: PathLike, save_dir: PathLike):
    """Convert from DICOM or nrrd to nifti"""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"{data_dir} does not exist")
    data = list(data_dir.glob("*"))[:-1]
    if len(data) == 0:
        raise ValueError(f"{data_dir} is empty")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    io_pipeline = Compose(
        [
            LoadImage(ensure_channel_first=True, reader="ITKReader"),
            SaveImage(output_dir=save_dir, output_postfix=""),
        ]
    )
    io_pipeline(data)


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
    input_dir: PathLike,
    output_dir: PathLike,
    out_filename: Optional[str] = None,
):
    """
    Args:
        input_dir: absolute path to the directory with all the cases containing
                dicoms
        output_dir: absolute path to the directory where to save nifties
    """
    for dicom_path in Path(input_dir).iterdir():
        if not dicom_path.is_dir():
            log.warning("Found additional files in the directory!")
        else:
            save_dir = Path(output_dir) / dicom_path.name
            save_dir.mkdir(exist_ok=True, parents=True)
            converter = get_dcm2niix_converter(
                dicom_path, save_dir, out_filename
            )
            converter.run()


@nrrd_app.command()
def nrrd_to_nifti(nrrd_path: str, output_path: str):
    """
    Converts an image in NRRD format to NifTI format.
    """
    img = sitk.ReadImage(nrrd_path)
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    sitk.WriteImage(img, str(output_path))
