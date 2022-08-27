import abc
from pathlib import Path
from typing import Any

import pandas as pd
from monai.transforms import LoadImage, SaveImage


def save_image_as_nifti_for_pyradiomics(filepath: str, tmp_dir) -> str:
    """pyRadiomics requires nifti file.
    Load image in any format (dicom/nifti/..)
    and save if as nifti in a temporary directory.

    Args:
       filepath: path to the image file / DICOM directory
    """
    image, meta = LoadImage()(filepath)
    out_dir = Path(tmp_dir)
    SaveImage(output_dir=out_dir)(image, meta)

    out_fname = Path(filepath).stem.split(".")[0] + "_trans.nii.gz"
    out_path = out_dir / out_fname

    return str(out_path)


class AbstractFeatureExtractor(abc.ABC):
    """Abstract class for all the feature extractors.
    Every extractor implementation should extend this abstract class
    and implement the methods marked as abstract: execute, describe.
    """

    def __init__(self, filepath: str, tmp_dir: str, verbose: bool = False):
        """
        Args:
            filepath: path to the image file / DICOM directory
            tmp_dir: path to the temporary directory
            verbose: logging for pyradiomics
        """
        self.filepath = filepath
        self.tmp_dir = tmp_dir
        self.verbose = verbose

    @abc.abstractmethod
    def execute(self) -> pd.DataFrame:
        """
        Returns:
            Dataframe containing the extracted features
        """
        pass

    @abc.abstractmethod
    def describe(self) -> dict[str, Any]:
        """
        Returns:
            A dict containing the attributes of the feature extractor
        """
        pass
