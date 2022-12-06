import abc
from pathlib import Path
from typing import Any

import pandas as pd


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
