import functools
import json
import logging
import os
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
import pydicom_seg
import pydicom

log = logging.getLogger(__name__)


def read_dicom_sitk(input_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(input_dir))
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image = sitk.DICOMOrient(image, "LPS")

    return image


def read_image_sitk(input_path: Path) -> sitk.Image:
    if input_path.is_dir():
        vol = read_dicom_sitk(input_path)
    else:
        vol = sitk.ReadImage(str(input_path))

    return vol


def read_dicom_seg_sitk(input_path: Path, overlapping=False, label: int | None = None) -> sitk.Image:
    """
    Args:
        input_path (Path): path to the DICOM SEG file
        overlapping (bool, optional): Use if segmentation segments overlap
            (e.g. prostate and prostate lesion). Defaults to False.
        label (int, optional): Specify which label to read. Used only
            for overlapping segmentation.

    Raises:
        FileNotFoundError: _description_
        ValueError: _description_

    Returns:
        sitk.Image: _description_
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"DICOM SEG file not found at {input_path}")
    dcm = pydicom.dcmread(str(input_path))
    if not overlapping:
        reader = pydicom_seg.MultiClassReader()
        seg = reader.read(dcm).image
    else:
        reader = pydicom_seg.SegmentReader()
        if label is None:
            raise ValueError("Label must be specified for overlapping segmentation")
        seg = reader.read(dcm).segment_image(label)
    return seg


def read_segmentation_sitk(input_path: Path, label=None) -> sitk.Image:
    if input_path.is_dir():
        files = list(input_path.glob("*.dcm"))
        if len(files) != 1:
            raise ValueError(f"Failed to read segmentation from {input_path}")
        input_path = files[0]
    if input_path.name.endswith(".dcm"):
        if label is not None:
            seg = read_dicom_seg_sitk(input_path, overlapping=True, label=label)
        else:
            seg = read_dicom_seg_sitk(input_path)
    else:
        seg = sitk.ReadImage(str(input_path))
    return seg


def load_yaml(yaml_path):
    """
    Reads .yaml file and returns a dictionary
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data


def save_yaml(data, yaml_path):
    """
    Saves a dictionary to .yaml file
    """
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data


def save_json(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def load_sitk(img_path) -> sitk.Image:
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"File not found at {img_path}")
    img = sitk.ReadImage(str(img_path))
    return img


def save_sitk(img, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(output_path))


def get_sitk_array(img: sitk.Image) -> np.ndarray:
    """
    Convert a SimpleITK image to a NumPy array.

    Args:
        img: The SimpleITK image to convert.

    Returns:
        A NumPy array with the same data as the input image.
    """
    # Get the array from the SimpleITK image and reorder the dimensions
    # to [depth, height, width]
    arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0)
    return arr


def load_array(img_path) -> np.ndarray:
    img = load_sitk(img_path)
    arr = get_sitk_array(img)
    return arr


def load_nibabel(img_path) -> nib.Nifti1Image:
    img_path = Path(img_path)
    if not img_path.exists():
        raise FileNotFoundError(f"File not found at {img_path}")
    img = nib.load(str(img_path))
    return img


def save_nibabel(img, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, output_path)


def save_predictions_to_csv(y_true, y_pred, output_path):
    predictions = pd.DataFrame(
        {"y_true": y_true, "y_pred_proba": y_pred}
    ).sort_values("y_true", ascending=False)
    predictions.to_csv(output_path, index=False)


def nifti_io(func):
    @functools.wraps(func)
    def wrapper(
        input_path: Path | str, output_path: Path | str, *args, **kwargs
    ):
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Nifti not found at {str(input_path)}.")
        nifti = nib.load(input_path)

        # Call the original function with the nifti image as the first argument
        result = func(nifti, *args, **kwargs)

        # Save the nifti image
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(result, output_path)
        log.info(f"Saved mask to {str(output_path)}.")

        return result

    return wrapper


def zip_directory(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, mode="w") as zipf:
        len_dir_path = len(folder_path)
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, file_path[len_dir_path:])
