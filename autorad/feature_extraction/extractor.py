import logging
from pathlib import Path

import mlflow
import pandas as pd
from joblib import Parallel, delayed
from radiomics import featureextractor
from tqdm import tqdm

from autorad.config import config
from autorad.config.type_definitions import PathLike
from autorad.data.dataset import ImageDataset
from autorad.utils import io, mlflow_utils, utils

log = logging.getLogger(__name__)

# Silence the pyRadiomics logger
logging.getLogger("radiomics").setLevel(logging.WARNING)


class FeatureExtractor:
    def __init__(
        self,
        dataset: ImageDataset,
        feature_set: str = "pyradiomics",
        extraction_params: PathLike = "CT_Baessler.yaml",
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
        self.n_jobs = utils.set_n_jobs(n_jobs)
        self._initialize_extractor()

    def _get_extraction_param_path(self, extraction_params: PathLike) -> str:
        default_extraction_param_dir = Path(config.PARAM_DIR)
        if Path(extraction_params).is_file():
            result_path = Path(extraction_params)
        elif (default_extraction_param_dir / str(extraction_params)).is_file():
            result_path = default_extraction_param_dir / str(extraction_params)
        else:
            raise ValueError(
                f"Extraction parameter file {extraction_params} not found."
            )
        return str(result_path)

    def run(self, keep_metadata=True) -> pd.DataFrame:
        """
        Run feature extraction.
        Args:
            keep_metadata: merge extracted features with data from the
            ImageDataset.df.
        Returns:
            DataFrame containing extracted features
        """
        log.info("Extracting features")
        if self.n_jobs is None or self.n_jobs == 1:
            feature_df = self.get_features()
        else:
            feature_df = self.get_features_parallel()

        ID_colname = self.dataset.ID_colname
        feature_df = feature_df.astype(float)
        feature_df.insert(0, ID_colname, self.dataset.ids)

        run_id = self.save_config()

        # add ID for this extraction run
        feature_df.insert(1, "extraction_ID", run_id)

        if keep_metadata:
            # Add all columns from ImageDataset.df
            try:
                feature_df = self.dataset.df.merge(
                    feature_df,
                    on=ID_colname,
                )
            except ValueError:
                raise ValueError("Error concatenating features and metadata.")
        return feature_df

    def save_config(self):
        extraction_param_dict = io.load_yaml(self.extraction_params)
        run_config = {
            "feature_set": self.feature_set,
            "extraction_params": extraction_param_dict,
        }

        mlflow.set_tracking_uri("file://" + config.MODEL_REGISTRY)
        mlflow.set_experiment("feature_extraction")
        with mlflow.start_run() as run:
            mlflow_utils.log_dict_as_artifact(run_config, "extraction_config")

        return run.info.run_id

    def _initialize_extractor(self):
        if self.feature_set == "pyradiomics":
            self.extractor = PyRadiomicsExtractorWrapper(
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
            return None

        return feature_dict

    @utils.time_it
    def get_features(self) -> pd.DataFrame:
        """
        Get features for all cases.
        """
        lst_of_feature_dicts = [
            self.get_features_for_single_case(image_path, mask_path)
            for image_path, mask_path in tqdm(
                zip(self.dataset.image_paths, self.dataset.mask_paths)
            )
        ]
        feature_df = pd.DataFrame(lst_of_feature_dicts)
        return feature_df

    @utils.time_it
    def get_features_parallel(self) -> pd.DataFrame:
        with Parallel(n_jobs=self.n_jobs) as parallel:
            lst_of_feature_dicts = parallel(
                delayed(self.get_features_for_single_case)(
                    image_path, mask_path
                )
                for image_path, mask_path in zip(
                    self.dataset.image_paths, self.dataset.mask_paths
                )
            )
        feature_df = pd.DataFrame(lst_of_feature_dicts)
        return feature_df

    def get_pyradiomics_feature_names(self) -> list[str]:
        class_obj = featureextractor.getFeatureClasses()
        feature_classes = list(class_obj.keys())
        feature_names = [
            f"{klass}_{name}"
            for klass in feature_classes
            for name in class_obj[klass].getFeatureNames().keys()
        ]
        return feature_names


class PyRadiomicsExtractorWrapper(featureextractor.RadiomicsFeatureExtractor):
    """Wrapper that filters out extracted metadata"""

    def __init__(self, extraction_params: PathLike, *args, **kwargs):
        super().__init__(str(extraction_params), *args, **kwargs)

    def execute(self, image_path: PathLike, mask_path: PathLike) -> dict:
        feature_dict = dict(super().execute(str(image_path), str(mask_path)))
        feature_dict_without_metadata = {
            feature_name: feature_dict[feature_name]
            for feature_name in feature_dict.keys()
            if "diagnostic" not in feature_name
        }
        return feature_dict_without_metadata
