from __future__ import annotations

import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from radiomics import featureextractor
from tqdm import tqdm

from classrad.config import config
from classrad.config.type_definitions import PathLike
from classrad.data.dataset import ImageDataset
from classrad.utils.utils import time_it


class FeatureExtractor:
    def __init__(
        self,
        dataset: ImageDataset,
        out_path: PathLike,
        feature_set: str = "pyradiomics",
        extraction_params: PathLike = "Baessler_CT.yaml",
        verbose: bool = False,
    ):
        """
        Args:
            - dataset: ImageDataset containing image paths, mask paths, and IDs
            - out_path: Path to save feature dataframe
            - feature_set: library to use features from (for now only pyradiomics)
            - extraction_params: path to the JSON file containing the extraction
                parameters, or a string containing the name of the file in the
                default extraction parameter directory.
            - verbose: logging mainly for pyradiomics
        """
        self.dataset = dataset
        self.out_path = out_path
        self.feature_set = feature_set
        self.extraction_params = self._get_extraction_param_path(
            extraction_params
        )
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.info("FeatureExtractor initialized")

    @staticmethod
    def _get_extraction_param_path(extraction_params: PathLike) -> Path:
        default_extraction_param_dir = Path(config.PARAM_DIR)
        if Path(extraction_params).is_file():
            result = Path(extraction_params)
        elif (default_extraction_param_dir / str(extraction_params)).is_file():
            result = default_extraction_param_dir / extraction_params
        else:
            raise ValueError(
                f"Extraction parameter file {extraction_params} not found."
            )
        return result

    def extract_features(self, num_threads: int = 1):
        """
        Extract features from a set of images.
        """
        # Get the feature extractor
        self.logger.info("Initializing feature extractor")
        self.logger.info(
            f"Using extraction params from {self.extraction_params}"
        )
        self._initialize_extractor()

        # Get the feature values
        self.logger.info("Extracting features")
        if num_threads > 1:
            feature_df = self.get_features_parallel(num_threads)
        else:
            feature_df = self.get_features()
        feature_df.to_csv(self.out_path, index=False)

    def _initialize_extractor(self):
        """
        Initialize feature extractor.
        """
        if self.feature_set == "pyradiomics":
            self.extractor = featureextractor.RadiomicsFeatureExtractor(
                str(self.extraction_params)
            )
        else:
            raise ValueError("Feature set not supported")
        return self

    def _add_features_for_single_case(
        self, case: pd.Series
    ) -> pd.Series | None:
        """
        Run extraction for one case and append results to feature_df
        Args:
            case: a single row of the dataset.df
        """
        image_path = case[self.dataset.image_colname]
        mask_path = case[self.dataset.mask_colname]
        if not Path(image_path).is_file():
            self.logger.warning(
                f"Image not found. Skipping case... (path={image_path}"
            )
            return None
        if not Path(mask_path).is_file():
            self.logger.warning(
                f"Mask not found. Skipping case... (path={mask_path}"
            )
            return None
        feature_vector = self.extractor.execute(image_path, mask_path)
        # copy the all the metadata for the case
        feature_series = pd.concat([case, pd.Series(feature_vector)])

        return feature_series

    def get_feature_names(
        self, image_path: PathLike, mask_path: PathLike
    ) -> list[str]:
        """Get names of features from running it on the first case"""
        feature_vector = self.extractor.execute(image_path, mask_path)
        feature_names = list(feature_vector.keys())
        return feature_names

    def _initialize_feature_df(self):
        first_df_row = self.dataset.df.iloc[0]
        image_path = first_df_row[self.dataset.image_colname]
        mask_path = first_df_row[self.dataset.mask_colname]
        feature_names = self.get_feature_names(str(image_path), str(mask_path))
        feature_df = self.dataset.df.copy().reindex(columns=feature_names)
        return feature_df

    @time_it
    def get_features(self):
        feature_df = self._initialize_feature_df()
        rows = self.dataset.df.iterrows()
        for _, row in tqdm(rows):
            feature_series = self._add_features_for_single_case(row)
            if feature_series is not None:
                feature_df = feature_df.append(
                    feature_series, ignore_index=True
                )
        return feature_df

    @time_it
    def get_features_parallel(self, num_threads: int):
        feature_df = self._initialize_feature_df()
        try:
            _, df_rows = zip(*self.dataset.df.iterrows())
            p = Pool(num_threads)
            results = p.map(self._add_features_for_single_case, df_rows)
            feature_df = pd.concat(results, axis=1).T
        except Exception:
            print("Multiprocessing failed! :/")
        return feature_df
