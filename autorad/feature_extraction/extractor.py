import logging
import os
from pathlib import Path

import pandas as pd
import radiomics
from joblib import Parallel, delayed
from radiomics.featureextractor import RadiomicsFeatureExtractor
from tqdm import tqdm

from autorad.config import config
from autorad.config.type_definitions import PathLike
from autorad.data.dataset import ImageDataset
from autorad.utils.utils import time_it

log = logging.getLogger(__name__)
# Silence the pyRadiomics logger
logging.getLogger("radiomics").setLevel(logging.WARNING)


class FeatureExtractor:
    def __init__(
        self,
        dataset: ImageDataset,
        feature_set: str = "pyradiomics",
        extraction_params: PathLike = "Baessler_CT.yaml",
        n_jobs: int | None = None,
    ):
        """
        Args:
            dataset: ImageDataset containing image paths, mask paths, and IDs
            feature_set: library to use features from (for now only pyradiomics)
            extraction_params: path to the JSON file containing the extraction
                parameters, or a string containing the name of the file in the
                default extraction parameter directory
                (autorad.config.pyradiomics_params)
            n_jobs: number of parallel jobs to run
        Returns:
            None
        """
        self.dataset = dataset
        self.feature_set = feature_set
        self.extraction_params = self._get_extraction_param_path(
            extraction_params
        )
        log.info(f"Using extraction params from {self.extraction_params}")
        if n_jobs == -1:
            self.n_jobs = os.cpu_count()
        else:
            self.n_jobs = n_jobs
        self._initialize_extractor()

    def _get_extraction_param_path(self, extraction_params: PathLike) -> Path:
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

    def run(self) -> pd.DataFrame:
        """
        Run feature extraction.
        Returns a DataFrame with extracted features and metadata from the
        ImageDataset.
        """
        log.info("Extracting features")
        if self.n_jobs is None:
            feature_df = self.get_features()
        else:
            feature_df = self.get_features_parallel()

        # Add metadata
        try:
            result = feature_df.merge(
                self.dataset.df, left_index=True, right_index=True
            )
        except ValueError:
            raise ValueError("Error concatenating features and metadata.")
        return result

    def _initialize_extractor(self):
        if self.feature_set == "pyradiomics":
            self.extractor = RadiomicsFeatureExtractor(
                str(self.extraction_params)
            )
        else:
            raise ValueError("Feature set not supported")
        log.info(f"Initialized extractor {self.feature_set}")
        return self

    def get_features_for_single_case(
        self, image_path: PathLike, mask_path: PathLike
    ) -> dict | None:
        """
        Returns:
            feature_series: dict with extracted features
        """
        if not Path(image_path).is_file():
            log.warning(
                f"Image not found. Skipping case... (path={image_path}"
            )
            return None
        if not Path(mask_path).is_file():
            log.warning(f"Mask not found. Skipping case... (path={mask_path}")
            return None
        try:
            feature_dict = self.extractor.execute(
                str(image_path),
                str(mask_path),
            )
        except ValueError:
            error_msg = f"Error extracting features for image, \
                mask pair {image_path}, {mask_path}"
            log.error(error_msg)
            raise ValueError(error_msg)

        # feature_series = pd.Series(feature_vector)
        return dict(feature_dict)

    @time_it
    def get_features(self) -> pd.DataFrame:
        """
        Run extraction for all cases.
        """
        image_paths = self.dataset.image_paths
        mask_paths = self.dataset.mask_paths
        lst_of_feature_dicts = [
            self.get_features_for_single_case(image_path, mask_path)
            for image_path, mask_path in tqdm(zip(image_paths, mask_paths))
        ]
        feature_df = pd.DataFrame(lst_of_feature_dicts)

        return feature_df

    @time_it
    def get_features_parallel(self) -> pd.DataFrame:
        image_paths = self.dataset.image_paths
        mask_paths = self.dataset.mask_paths
        try:
            with Parallel(n_jobs=self.n_jobs) as parallel:
                list_of_feature_dicts = parallel(
                    delayed(self.get_features_for_single_case)(
                        image_path, mask_path
                    )
                    for image_path, mask_path in zip(image_paths, mask_paths)
                )
        except Exception:
            raise RuntimeError("Multiprocessing failed! :/")
        feature_df = pd.DataFrame(list_of_feature_dicts)
        return feature_df

    def get_pyradiomics_feature_names(self) -> list[str]:
        class_obj = radiomics.featureextractor.getFeatureClasses()
        feature_classes = list(class_obj.keys())
        feature_names = [
            name
            for klass in feature_classes
            for name in class_obj[klass].getFeatureNames().keys()
        ]
        return feature_names
