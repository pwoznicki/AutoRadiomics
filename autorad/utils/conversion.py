import logging
from pathlib import Path

import SimpleITK as sitk

log = logging.getLogger(__name__)


# def convert_dataset_to_nifti(data_dir: PathLike, save_dir: PathLike):
#     """Convert from DICOM or nrrd to nifti"""
#     data_dir = Path(data_dir)
#     if not data_dir.exists():
#         raise ValueError(f"{data_dir} does not exist")
#     data = list(data_dir.glob("*"))[:-1]
#     if len(data) == 0:
#         raise ValueError(f"{data_dir} is empty")
#     save_dir = Path(save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)
#     io_pipeline = Compose(
#         [
#             LoadImage(ensure_channel_first=True, reader="ITKReader"),
#             SaveImage(output_dir=save_dir, output_postfix=""),
#         ]
#     )
#     io_pipeline(data)


def convert_to_nifti(input_path: Path, output_path: Path):
    """Load image in any format (dicom/nifti/..)
    and save if as nifti in a temporary directory using SimpleITK.

    Args:
        input_path: path to the image file / DICOM directory
        output_path: path to the output nifti file
    """
    img = sitk.ReadImage(str(input_path))
    sitk.WriteImage(img, str(output_path))


def nrrd_to_nifti(nrrd_path: str, output_path: str):
    """
    Converts an image in NRRD format to NifTI format.
    """
    img = sitk.ReadImage(nrrd_path)
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    sitk.WriteImage(img, str(output_path))
