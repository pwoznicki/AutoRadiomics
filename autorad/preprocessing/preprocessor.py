from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autorad.config import config, type_definitions
from autorad.data.dataset import TrainingData, TrainingInput, TrainingLabels
from autorad.feature_selection.selector import create_feature_selector

log = logging.getLogger(__name__)


def get_not_none_kwargs(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}


class Preprocessor:
    def __init__(
        self,
        standardize: bool = True,
        feature_selection_method: str | None = None,
        n_features: int | None = None,
        oversampling_method: str | None = None,
        random_state: int = config.SEED,
    ):
        """Performs preprocessing, including:
        1. standardization
        2. feature selection
        3. oversampling

        Args:
            standardize: whether to standardize features to mean 0 and std 1
            feature_selection_method: algorithm to select key features,
                if None, select all features
            n_features: number of features to select, only applicable to selected
                feature selection methods (see feature_selection.selector)
            oversampling_method: minority class oversampling method,
                if None, no oversampling
            random_state: seed
        """
        self.standardize = standardize
        self.feature_selection_method = feature_selection_method
        self.n_features = n_features
        self.oversampling_method = oversampling_method
        self.random_state = random_state
        self.pipeline = self._build_pipeline()
        # self.selected_features = None

    def fit_transform(self, data: TrainingData):
        # copy data
        _data = dataclasses.replace(data)
        X, y = _data.X, _data.y
        result_X = {}
        result_y = {}
        transformed = self.pipeline.fit_transform(X.train, y.train)
        if isinstance(transformed, tuple):
            result_X["train"], result_y["train"] = transformed
        else:
            result_X["train"] = transformed
            result_y["train"] = y.train
        result_X["test"] = self.pipeline.transform(X.test)
        result_y["test"] = y.test
        if X.val is not None:
            result_X["val"] = self.pipeline.transform(X.val)
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
            result_df_X_train, result_y_train = cv_pipeline.fit_transform(
                X_train, y_train
            )
            result_df_X_val = cv_pipeline.transform(X_val)
            result_X_train_folds.append(result_df_X_train)
            result_y_train_folds.append(result_y_train)
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
        if self.standardize:
            steps.append(
                (
                    "standardize",
                    StandardScaler().set_output(transform="pandas"),
                )
            )
        if self.feature_selection_method is not None:
            steps.append(
                (
                    "select",
                    create_feature_selector(
                        method=self.feature_selection_method,
                        **get_not_none_kwargs(n_features=self.n_features),
                        #    )
                    ),
                )
            )
        if self.oversampling_method is not None:
            steps.append(
                (
                    "oversample",
                    OversamplerWrapper(
                        create_oversampling_model(
                            method=self.oversampling_method,
                            random_state=self.random_state,
                        )
                    ),
                )
            )
        pipeline = Pipeline(steps)
        return pipeline


def run_auto_preprocessing(
    data: TrainingData,
    result_dir: type_definitions.PathLike,
    oversampling: bool = True,
    feature_selection: bool = True,
    selection_methods: list[str] | str = "all",
):
    if not feature_selection:
        selection_setups = [None]
    elif selection_methods == "all":
        selection_setups = config.FEATURE_SELECTION_METHODS
    else:
        selection_setups = selection_methods

    if oversampling:
        oversampling_methods = config.OVERSAMPLING_METHODS
    else:
        oversampling_methods = [None]

    preprocessed = {}
    for selection_method in selection_setups:
        preprocessed[selection_method] = {}
        for oversampling_method in oversampling_methods:
            preprocessor = Preprocessor(
                standardize=True,
                feature_selection_method=selection_method,
                oversampling_method=oversampling_method,
            )
            try:
                preprocessed[selection_method][
                    oversampling_method
                ] = preprocessor.fit_transform(data)
            except AssertionError:
                log.error(
                    f"Preprocessing with {selection_method} and {oversampling_method} failed."
                )
        if not preprocessed[selection_method]:
            del preprocessed[selection_method]
    with open(Path(result_dir) / "preprocessed.pkl", "wb") as f:
        joblib.dump(preprocessed, f)


def create_oversampling_model(method: str, random_state: int = config.SEED):
    if method is None:
        return None
    if method == "ADASYN":
        return ADASYN(random_state=random_state)
    elif method == "SMOTE":
        return SMOTE(random_state=random_state)
    elif method == "BorderlineSMOTE":
        return BorderlineSMOTE(random_state=random_state, kind="borderline1")
    raise ValueError(f"Unknown oversampling method: {method}")


class OversamplerWrapper:
    def __init__(self, oversampler, random_state=config.SEED):
        self.oversampler = oversampler
        self.oversampler.__init__(random_state=random_state)

    def fit(self, X, y):
        return self.oversampler.fit(X, y)

    def fit_transform(self, X, y):
        return self.oversampler.fit_resample(X, y)

    def transform(self, X):
        log.debug(f"{self.oversampler} does nothing on .transform()...")
        return X


class ScalerWrapper:
    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, X):
        return self.scaler.fit(X)

    def fit_transform(self, X, y=None):
        if y is None:
            return self.scaler.fit_transform(X)
        return self.scaler.fit_transform(X), y

    def transform(self, X, y=None):
        if y is None:
            return self.scaler.transform(X)
        return self.scaler.transform(X), y
