from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autorad.config import config
from autorad.data.dataset import TrainingData, TrainingInput, TrainingLabels
from autorad.feature_selection.selector import create_feature_selector

log = logging.getLogger(__name__)


class Preprocessor:
    def __init__(
        self,
        standardize: bool = True,
        feature_selection_method: str | None = None,
        oversampling_method: str | None = None,
        random_state: int = config.SEED,
        feature_selection_kwargs: dict[str, Any] | None = None,
    ):
        """Performs preprocessing, including:
        1. standardization
        2. feature selection
        3. oversampling

        Args:
            standardize: whether to standardize features to mean 0 and std 1
            feature_selection_method: algorithm to select key features,
                if None, don't perform selection and leave all features
            oversampling_method: minority class oversampling method,
                if None, no oversampling
            random_state: seed
            feature_selection_kwargs: keyword arguments for feature selection, e.g.
                {"n_features": 10} for `anova` method
        """
        self.standardize = standardize
        self.feature_selection_method = feature_selection_method
        self.oversampling_method = oversampling_method
        self.random_state = random_state
        if feature_selection_kwargs is None:
            self.feature_selection_kwargs = {}
        self.pipeline = self._build_pipeline()

    def fit_transform_data(self, data: TrainingData) -> TrainingData:
        _data = dataclasses.replace(data)
        X, y = _data.X, _data.y
        _data._X_preprocessed, _data._y_preprocessed = self.fit_transform(X, y)
        return _data

    def fit_transform(
        self, X: TrainingInput, y: TrainingLabels
    ) -> tuple[TrainingInput, TrainingLabels]:

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
            ) = self._fit_transform_cv_folds(X, y)
        X_preprocessed = TrainingInput(**result_X)
        y_preprocessed = TrainingLabels(**result_y)
        return X_preprocessed, y_preprocessed

    def _fit_transform_cv_folds(
        self, X: TrainingInput, y: TrainingLabels
    ) -> tuple[
        list[pd.DataFrame],
        list[pd.Series],
        list[pd.DataFrame],
        list[pd.Series],
    ]:
        if (
            X.train_folds is None
            or y.train_folds is None
            or X.val_folds is None
            or y.val_folds is None
        ):
            raise AttributeError("Folds are not set")
        (
            result_X_train_folds,
            result_y_train_folds,
            result_X_val_folds,
            result_y_val_folds,
        ) = ([], [], [], [])
        for X_train, y_train, X_val in zip(
            X.train_folds,
            y.train_folds,
            X.val_folds,
        ):
            cv_pipeline = self._build_pipeline()
            transformed = cv_pipeline.fit_transform(X_train, y_train)

            if isinstance(transformed, tuple):
                result_df_X_train, result_y_train = transformed
            else:
                result_df_X_train = transformed
                result_y_train = y_train

            result_df_X_val = cv_pipeline.transform(X_val)

            result_X_train_folds.append(result_df_X_train)
            result_y_train_folds.append(result_y_train)
            result_X_val_folds.append(result_df_X_val)
        result_y_val_folds = y.val_folds
        return (
            result_X_train_folds,
            result_y_train_folds,
            result_X_val_folds,
            result_y_val_folds,
        )

    def transform(self, X: TrainingInput):
        result_X = {}
        result_X["train"] = self.pipeline.transform(X.train)
        result_X["test"] = self.pipeline.transform(X.test)
        if X.val is not None:
            result_X["val"] = self.pipeline.transform(X.val)
        if X.train_folds is not None and X.val_folds is not None:
            (
                result_X["train_folds"],
                result_X["val_folds"],
            ) = self._transform_cv_folds(X)
        X_preprocessed = TrainingInput(**result_X)
        return X_preprocessed

    def _transform_cv_folds(
        self, X: TrainingInput
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        if X.train_folds is None or X.val_folds is None:
            raise AttributeError("Folds are not set")
        (
            result_X_train_folds,
            result_X_val_folds,
        ) = ([], [])
        for X_train, X_val in zip(
            X.train_folds,
            X.val_folds,
        ):
            result_df_X_train = self.pipeline.transform(X_train)
            result_df_X_val = self.pipeline.transform(X_val)
            result_X_train_folds.append(result_df_X_train)
            result_X_val_folds.append(result_df_X_val)
        return (
            result_X_train_folds,
            result_X_val_folds,
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
                        **self.feature_selection_kwargs,
                    ),
                ),
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
    result_dir: Path,
    use_oversampling: bool = True,
    use_feature_selection: bool = True,
    oversampling_methods: list[str] = None,
    feature_selection_methods: list[str] = None,
):
    """Run preprocessing with a variety of feature selection and oversampling methods.

    Args:
    - data: Training data to preprocess.
    - result_dir: Path to a directory where the preprocessed data will be saved.
    - use_oversampling: A boolean indicating whether to use oversampling. If `True` and
      `oversampling_methods` is not provided, all methods in the `config.OVERSAMPLING_METHODS`
      list will be used.
    - use_feature_selection: A boolean indicating whether to use feature selection. If `True` and
      `feature_selection_methods` is not provided, all methods in the `config.FEATURE_SELECTION_METHODS`
    - oversampling_methods: A list of oversampling methods to use. If not provided, all methods
      in the `config.OVERSAMPLING_METHODS` list will be used.
    - feature_selection_methods: A list of feature selection methods to use. If not provided, all
      methods in the `config.FEATURE_SELECTION_METHODS` list will be used.

    Returns:
    - None. The preprocessed data will be saved to the `result_dir` directory.
    """
    if use_oversampling:
        if oversampling_methods is None:
            oversampling_methods = config.OVERSAMPLING_METHODS
    else:
        oversampling_methods = [None]

    if use_feature_selection:
        if feature_selection_methods is None:
            feature_selection_methods = config.FEATURE_SELECTION_METHODS
    else:
        feature_selection_methods = []

    preprocessed = {}
    for selection_method in feature_selection_methods:
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
                ] = preprocessor.fit_transform_data(data)
            except AssertionError:
                log.error(
                    f"Preprocessing failed with {selection_method} and {oversampling_method}."
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
