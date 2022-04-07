from __future__ import annotations

import dataclasses
import logging

import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from classrad.config import config
from classrad.data.dataset import TrainingData, TrainingInput, TrainingLabels
from classrad.feature_selection.feature_selector import FeatureSelector

log = logging.getLogger(__name__)


class Preprocessor:
    def __init__(
        self,
        normalize: bool = True,
        feature_selection_method: str | None = None,
        n_features: int = 10,
        oversampling_method: str | None = None,
        random_state: int = config.SEED,
    ):
        self.normalize = normalize
        self.feature_selection_method = feature_selection_method
        self.n_features = n_features
        self.oversampling_method = oversampling_method
        self.random_state = random_state
        self.pipeline = self._build_pipeline()
        self.selected_features = None

    def transform(self, X: pd.DataFrame):
        result_array = self.pipeline.transform(X)
        result_df = pd.DataFrame(result_array, columns=self.selected_features)
        return result_df

    def fit_transform(self, data: TrainingData):
        # copy data
        _data = dataclasses.replace(data)
        X, y = _data.X, _data.y
        result_X = {}
        result_y = {}
        all_features = X.train.columns.tolist()
        X_train_trans, y_train_trans = self.pipeline.fit_transform(
            X.train, y.train, select__column_names=all_features
        )
        self.selected_features = self.pipeline["select"].selected_features
        result_X["train"] = pd.DataFrame(
            X_train_trans, columns=self.selected_features
        )
        result_y["train"] = pd.Series(y_train_trans)
        X_test_trans = self.pipeline.transform(X.test)
        result_X["test"] = pd.DataFrame(
            X_test_trans, columns=self.selected_features
        )
        result_y["test"] = y.test
        if X.val is not None:
            X_val_trans = self.pipeline.transform(X.val)
            result_X["val"] = pd.DataFrame(
                X_val_trans, columns=self.selected_features
            )
            result_y["val"] = y.val
        if X.train_folds is not None and X.val_folds is not None:
            (
                result_X["train_folds"],
                result_y["train_folds"],
                result_X["val_folds"],
                result_y["val_folds"],
            ) = self._fit_transform_cv_folds(_data)
        _data._X_preprocessed = TrainingInput(**result_X)
        _data._y_preprocessed = TrainingLabels(**result_y)
        return _data

    def _fit_transform_cv_folds(
        self, data: TrainingData
    ) -> tuple[
        list[pd.DataFrame],
        list[pd.Series],
        list[pd.DataFrame],
        list[pd.Series],
    ]:
        if (
            data.X.train_folds is None
            or data.y.train_folds is None
            or data.X.val_folds is None
            or data.y.val_folds is None
        ):
            raise AttributeError("Folds are not set")
        (
            result_X_train_folds,
            result_y_train_folds,
            result_X_val_folds,
            result_y_val_folds,
        ) = ([], [], [], [])
        for X_train, y_train, X_val in zip(
            data.X.train_folds,
            data.y.train_folds,
            data.X.val_folds,
        ):
            cv_pipeline = self._build_pipeline()
            all_features = X_train.columns.tolist()
            result_X_train, result_y_train = cv_pipeline.fit_transform(
                X_train, y_train, select__column_names=all_features
            )
            selected_features = cv_pipeline["select"].selected_features
            result_X_val = cv_pipeline.transform(X_val)
            result_df_X_train = pd.DataFrame(
                result_X_train, columns=selected_features
            )
            result_df_X_val = pd.DataFrame(
                result_X_val, columns=selected_features
            )
            result_X_train_folds.append(result_df_X_train)
            result_y_train_folds.append(pd.Series(result_y_train))
            result_X_val_folds.append(result_df_X_val)
        result_y_val_folds = data.y.val_folds
        return (
            result_X_train_folds,
            result_y_train_folds,
            result_X_val_folds,
            result_y_val_folds,
        )

    def _build_pipeline(self):
        steps = []
        if self.normalize:
            steps.append(("normalize", MinMaxScaler()))
        if self.feature_selection_method is not None:
            steps.append(
                (
                    "select",
                    FeatureSelector(
                        self.feature_selection_method, self.n_features
                    ),
                )
            )
        if self.oversampling_method is not None:
            steps.append(("balance", self._get_oversampling_model()))
        pipeline = Pipeline(steps)
        return pipeline

    def _get_oversampling_model(self):
        if self.oversampling_method is None:
            return None
        if self.oversampling_method == "ADASYN":
            return ADASYNWrapper(random_state=self.random_state)
        elif self.oversampling_method == "SMOTE":
            return SMOTEWrapper(random_state=self.random_state)
        elif self.oversampling_method == "BorderlineSMOTE":
            return BorderlineSMOTEWrapper(
                random_state=self.random_state, kind="borderline1"
            )
        raise ValueError(
            f"Unknown oversampling method: {self.oversampling_method}"
        )


class ADASYNWrapper(ADASYN):
    def __init__(self, random_state=config.SEED):
        super().__init__(random_state=random_state)

    def fit_transform(self, data, *args):
        return super().fit_resample(*data)

    def transform(self, X):
        log.info("ADASYN does notiong on .transform()...")
        return X


class SMOTEWrapper(SMOTE):
    def __init__(self, random_state=config.SEED):
        super().__init__(random_state=random_state)

    def fit_transform(self, data, *args):
        return super().fit_resample(*data)

    def transform(self, X):
        log.info("SMOTE does nothing on .transform()...")
        return X


class BorderlineSMOTEWrapper(BorderlineSMOTE):
    def __init__(self, kind="borderline-1", random_state=config.SEED):
        super().__init__(kind=kind, random_state=random_state)

    def fit_transform(self, data, *args):
        return super().fit_resample(*data)

    def transform(self, X):
        log.info("BorderlineSMOTE does nothing on .transform()...")
        return X
