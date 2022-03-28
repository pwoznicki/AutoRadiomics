from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from classrad.config import config
from classrad.data.dataset import TrainingData, TrainingInput, TrainingLabels
from classrad.feature_selection.feature_selector import FeatureSelector


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

    def fit_transform(self, data: TrainingData):
        X, y = data.X, data.y
        result_X = {}
        result_y = {}
        X_cols = X.train.columns.tolist()
        X_train_trans, y_train_trans = self.pipeline.fit_transform(
            data.X.train, data.y.train
        )
        result_X["train"] = pd.DataFrame(X_train_trans, columns=X_cols)
        result_y["train"] = pd.Series(y_train_trans)
        X_test_trans, y_test_trans = self.pipeline.transform(X.test, y.test)
        result_X["test"] = pd.DataFrame(X_test_trans, columns=X_cols)
        result_y["test"] = pd.Series(y_test_trans)
        if X.val is not None and y.val is not None:
            X_val_trans, y_val_trans = self.pipeline.transform(X.val)
            result_X["val"] = pd.DataFrame(X_val_trans, columns=X_cols)
            result_y["val"] = pd.Series(y_val_trans)
        if X.train_folds is not None and X.val_folds is not None:
            (
                result_X["train_folds"],
                result_y["train_folds"],
                result_X["val_folds"],
                result_y["val_folds"],
            ) = self._fit_transform_cv_folds(data)
        data._X_preprocessed = TrainingInput(**result_X)
        data._y_preprocessed = TrainingLabels(**result_y)
        return data

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
        for X_train, y_train, X_val, y_val in zip(
            data.X.train_folds,
            data.y.train_folds,
            data.X.val_folds,
            data.y.val_folds,
        ):
            result_X_train, result_y_train = self.pipeline.fit_transform(
                X_train, y_train
            )
            result_X_val, result_y_val = self.pipeline.transform(X_val, y_val)
            result_df_X_train = pd.DataFrame(
                result_X_train, columns=X_train.columns
            )
            result_df_X_val = pd.DataFrame(result_X_val, columns=X_val.columns)
            result_X_train_folds.append(result_df_X_train)
            result_y_train_folds.append(pd.Series(result_y_train))
            result_X_val_folds.append(result_df_X_val)
            result_y_val_folds.append(pd.Series(result_y_val))
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
            return ADASYN(random_state=self.random_state)
        elif self.oversampling_method == "SMOTE":
            return SMOTE(random_state=self.random_state)
        raise ValueError(
            f"Unknown oversampling method: {self.oversampling_method}"
        )
