import logging
from pathlib import Path

import SimpleITK as sitk
import typer
from monai.transforms import Compose, LoadImage, SaveImage

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


@nrrd_app.command()
def nrrd_to_nifti(nrrd_path: str, output_path: str):
    """
    Converts an image in NRRD format to NifTI format.
    """
    img = sitk.ReadImage(nrrd_path)
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    sitk.WriteImage(img, str(output_path))
