import logging
import tempfile
from pathlib import Path

import pandas as pd

from autorad.config.type_definitions import PathLike
from autorad.data import FeatureDataset, ImageDataset, TrainingData
from autorad.feature_extraction.extractor import FeatureExtractor
from autorad.utils import extraction_utils, io

log = logging.getLogger(__name__)


class Inferrer:
    def __init__(self, model, extraction_config, preprocessor, result_dir):
        self.model = model
        self.preprocessor = preprocessor
        self.extraction_config = extraction_config
        self.result_dir = result_dir

    def extract_features(
        self, img_path: PathLike, mask_path: PathLike
    ) -> pd.DataFrame:
        feature_df = infer_radiomics_features(
            img_path, mask_path, self.extraction_config
        )
        return feature_df

    def preprocess_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        return self.preprocessor.pipeline.transform(feature_df)

    def predict_proba_from_preprocessed(self, X: pd.DataFrame) -> float:
        y_pred_proba = self.model.predict_proba_binary(X)[0]
        return y_pred_proba

    def predict_proba(self, img_path: PathLike, mask_path: PathLike) -> float:
        feature_df = self.extract_features(img_path, mask_path)
        X = self.preprocess_features(feature_df)
        y_pred_proba = self.predict_proba_from_preprocessed(X)
        return y_pred_proba

    def predict_proba_with_features(
        self, img_path: PathLike, mask_path: PathLike
    ) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        feature_df = self.extract_features(img_path, mask_path)
        X = self.preprocess_features(feature_df)
        y_pred_proba = self.predict_proba_from_preprocessed(X)
        return (feature_df, X, y_pred_proba)

    # def fit(self, dataset: FeatureDataset):
    #     _data = self.preprocessor.fit_transform(dataset.data)
    #     self.model.fit(
    #         _data._X_preprocessed.train, _data._y_preprocessed.train
    #     )

    # def eval(self, dataset: FeatureDataset, result_name: str = "results"):
    #     X = self.preprocessor.transform(dataset.data.X.test)
    #     y = dataset.data.y.test
    #     y_pred = self.model.predict_proba_binary(X)
    #     auc = roc_auc_score(y, y_pred)
    #     result = {}
    #     result["selected_features"] = self.preprocessor.selected_features
    #     result["AUC test"] = auc
    #     io.save_json(result, (self.result_dir / f"{result_name}.json"))
    #     io.save_predictions_to_csv(
    #         y, y_pred, (self.result_dir / f"{result_name}.csv")
    #     )

    # def fit_eval(self, dataset: FeatureDataset, result_name: str = "results"):
    #     _data = self.preprocessor.fit_transform(dataset.data)
    #     result = {}
    #     result["selected_features"] = _data.selected_features
    #     train_auc = self._fit_eval_splits(_data)
    #     result["AUC train"] = train_auc
    #     test_auc = self._fit_eval_train_test(_data, result_name)
    #     result["AUC test"] = test_auc
    #     log.info(
    #         f"Test AUC: {test_auc:.3f}, mean train AUC: {np.mean(train_auc):.3f}"
    #     )
    #     io.save_json(result, (self.result_dir / f"{result_name}.json"))

    # def _fit_eval_train_test(
    #     self, _data: TrainingData, result_name: str = "results"
    # ):
    #     self.model.fit(
    #         _data._X_preprocessed.train, _data._y_preprocessed.train
    #     )
    #     y_pred = self.model.predict_proba_binary(_data._X_preprocessed.test)
    #     y_test = _data._y_preprocessed.test
    #     auc = roc_auc_score(y_test, y_pred)
    #     io.save_predictions_to_csv(
    #         y_test, y_pred, (self.result_dir / f"{result_name}.csv")
    #     )
    #     return auc

    # def _fit_eval_splits(self, data: TrainingData):
    #     aucs = []
    #     for X_train, y_train, _, X_val, y_val, _ in data.iter_training():
    #         self.model.fit(X_train, y_train)
    #         y_pred = self.model.predict_proba_binary(X_val)
    #         auc_val = roc_auc_score(y_val, y_pred)
    #         aucs.append(auc_val)

    #     return aucs

    def init_result_df(self, dataset: FeatureDataset):
        self.result_df = dataset.meta_df.copy()
        self.test_indices = dataset.X.test.index.values


def infer_radiomics_features(img_path, mask_path, extraction_config):
    if not Path(img_path).exists():
        raise FileNotFoundError(f"Image {img_path} does not exist")
    if not Path(mask_path).exists():
        raise FileNotFoundError(f"Mask {mask_path} does not exist")
    path_df = pd.DataFrame(
        {
            "image_path": [str(img_path)],
            "segmentation_path": [str(mask_path)],
        }
    )
    image_dataset = ImageDataset(
        path_df,
        image_colname="image_path",
        mask_colname="segmentation_path",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        extraction_param_path = Path(tmpdir) / "extraction_params.yaml"
        io.save_yaml(
            extraction_config["extraction_params"], extraction_param_path
        )
        extractor = FeatureExtractor(
            image_dataset,
            feature_set=extraction_config["feature_set"],
            extraction_params=extraction_param_path,
        )
        feature_df = extractor.run()
    radiomics_features = extraction_utils.filter_pyradiomics_names(
        list(feature_df.columns)
    )
    feature_df = feature_df[radiomics_features]
    return feature_df
