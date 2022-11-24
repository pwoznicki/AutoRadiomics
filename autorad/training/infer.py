import logging

import numpy as np
from sklearn.metrics import roc_auc_score

from autorad.data.dataset import FeatureDataset, TrainingData
from autorad.utils import io

log = logging.getLogger(__name__)


class Inferrer:
    def __init__(self, model, preprocessor, result_dir):
        self.result_dir = result_dir
        self.model = model
        self.preprocessor = preprocessor

    def fit(self, dataset: FeatureDataset):
        _data = self.preprocessor.fit_transform(dataset.data)
        self.model.fit(
            _data._X_preprocessed.train, _data._y_preprocessed.train
        )

    def predict(self, feature_df):
        X = self.preprocessor.pipeline.transform(feature_df)
        y_pred = self.model.predict(X)
        return y_pred

    def eval(self, dataset: FeatureDataset, result_name: str = "results"):
        X = self.preprocessor.transform(dataset.data.X.test)
        y = dataset.data.y.test
        y_pred = self.model.predict_proba_binary(X)
        auc = roc_auc_score(y, y_pred)
        result = {}
        result["selected_features"] = self.preprocessor.selected_features
        result["AUC test"] = auc
        io.save_json(result, (self.result_dir / f"{result_name}.json"))
        io.save_predictions_to_csv(
            y, y_pred, (self.result_dir / f"{result_name}.csv")
        )

    def fit_eval(self, dataset: FeatureDataset, result_name: str = "results"):
        _data = self.preprocessor.fit_transform(dataset.data)
        result = {}
        result["selected_features"] = _data.selected_features
        train_auc = self._fit_eval_splits(_data)
        result["AUC train"] = train_auc
        test_auc = self._fit_eval_train_test(_data, result_name)
        result["AUC test"] = test_auc
        log.info(
            f"Test AUC: {test_auc:.3f}, mean train AUC: {np.mean(train_auc):.3f}"
        )
        io.save_json(result, (self.result_dir / f"{result_name}.json"))

    def _fit_eval_train_test(
        self, _data: TrainingData, result_name: str = "results"
    ):
        self.model.fit(
            _data._X_preprocessed.train, _data._y_preprocessed.train
        )
        y_pred = self.model.predict_proba_binary(_data._X_preprocessed.test)
        y_test = _data._y_preprocessed.test
        auc = roc_auc_score(y_test, y_pred)
        io.save_predictions_to_csv(
            y_test, y_pred, (self.result_dir / f"{result_name}.csv")
        )
        return auc

    def _fit_eval_splits(self, data: TrainingData):
        aucs = []
        for X_train, y_train, _, X_val, y_val, _ in data.iter_training():
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict_proba_binary(X_val)
            auc_val = roc_auc_score(y_val, y_pred)
            aucs.append(auc_val)

        return aucs

    def init_result_df(self, dataset: FeatureDataset):
        self.result_df = dataset.meta_df.copy()
        self.test_indices = dataset.X.test.index.values
