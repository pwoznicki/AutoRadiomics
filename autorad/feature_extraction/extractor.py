import logging
from pathlib import Path

import mlflow
import pandas as pd
from pqdm.processes import pqdm
from radiomics import featureextractor
from tqdm import tqdm

from autorad.config import config
from autorad.config.type_definitions import PathLike
from autorad.data import ImageDataset
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

    def run(
        self, keep_metadata=True, mask_label: int | None = None
    ) -> pd.DataFrame:
        """
        Run feature extraction.
        Args:
            keep_metadata: merge extracted features with data from the
                ImageDataset.df.
            mask_label: label in the mask to extract features from.
                For default value of None, the `label` value from extraction
                param file is used. Set this when you have multiple labels in your mask
        Returns:
            DataFrame containing extracted features
        """
        log.info("Extracting features")
        if self.n_jobs is None or self.n_jobs == 1:
            feature_df = self.get_features(mask_label=mask_label)
        else:
            feature_df = self.get_features_parallel(mask_label=mask_label)
        if feature_df.empty:
            raise ValueError(
                "No features extracted. Check the logs and your dataset."
            )

        ID_colname = self.dataset.ID_colname
        # move ID column to front
        feature_df = feature_df.set_index(ID_colname).reset_index()

        run_id = self.save_config(mask_label=mask_label)

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

    def save_config(self, mask_label):
        extraction_param_dict = io.load_yaml(self.extraction_params)
        if mask_label is not None:
            extraction_param_dict["label"] = mask_label
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
        self,
        image_path: PathLike,
        mask_path: PathLike,
        ID: str | None = None,
        mask_label: int | None = None,
    ) -> dict | None:
        """
        Returns:
            feature_series: dict with extracted features
        """
        image_path = Path(image_path)
        mask_path = Path(mask_path)

        if not image_path.exists():
            log.warning(
                f"Image not found. Skipping case... (path={image_path}"
            )
            return None
        if not mask_path.exists():
            log.warning(f"Mask not found. Skipping case... (path={mask_path}")
            return None
        try:
            feature_dict = self.extractor.execute(
                image_path, mask_path, label=mask_label
            )
        except Exception as e:
            error_msg = f"Error extracting features for image, mask pair: {image_path}, {mask_path}"
            log.error(error_msg)
            log.error(f"Original error: {e}")
            return None

        if ID is not None:
            feature_dict[self.dataset.ID_colname] = ID

        return feature_dict

    @utils.time_it
    def get_features(self, mask_label=None) -> pd.DataFrame:
        """
        Get features for all cases.
        """
        lst_of_feature_dicts = [
            self.get_features_for_single_case(
                image_path, mask_path, id_, mask_label=mask_label
            )
            for image_path, mask_path, id_ in tqdm(
                list(
                    zip(
                        self.dataset.image_paths,
                        self.dataset.mask_paths,
                        self.dataset.ids,
                    )
                )
            )
        ]
        lst_of_feature_dicts = [
            feature_dict
            for feature_dict in lst_of_feature_dicts
            if feature_dict is not None
        ]
        feature_df = pd.DataFrame(lst_of_feature_dicts)
        return feature_df

    @utils.time_it
    def get_features_parallel(self, mask_label=None) -> pd.DataFrame:
        lst_of_feature_dicts = pqdm(
            (
                {
                    "image_path": vals[0],
                    "mask_path": vals[1],
                    "ID": vals[2],
                    "mask_label": mask_label,
                }
                for vals in zip(
                    self.dataset.image_paths,
                    self.dataset.mask_paths,
                    self.dataset.ids,
                )
            ),
            self.get_features_for_single_case,
            n_jobs=self.n_jobs,
            argument_type="kwargs",
        )
        lst_of_feature_dicts = [
            feature_dict
            for feature_dict in lst_of_feature_dicts
            if feature_dict is not None
        ]
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

    def execute(
        self,
        image_path: PathLike,
        mask_path: PathLike,
        label: int | None = None,
    ) -> dict:
        img = io.read_image_sitk(Path(image_path))
        mask = io.read_segmentation_sitk(Path(mask_path), label=label)
        feature_dict = dict(super().execute(img, mask, label=label))
        feature_dict_without_metadata = {
            feature_name: feature_dict[feature_name]
            for feature_name in feature_dict.keys()
            if "diagnostic" not in feature_name
        }
        return feature_dict_without_metadata
