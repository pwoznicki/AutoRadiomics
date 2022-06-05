from __future__ import annotations

import abc
import logging
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from boruta import BorutaPy
from BorutaShap import BorutaShap
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

from autorad.config import config

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
log = logging.getLogger(__name__)


class NoFeaturesSelectedError(Exception):
    """raised when feature selection fails"""

    pass


class CoreSelector(abc.ABC):
    """Template for feature selection methods"""

    def __init__(self):
        self.selected_columns = None

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> list[int]:
        """fit method should update self.selected_columns.
        If no features are selected, it should raise
        NoFeaturesSelectedError.
        """
        pass

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.fit(X, y)
        return X[:, self.selected_columns], y

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_columns is None:
            raise NoFeaturesSelectedError(
                "No features selected!" "Call fit() first before transforming."
            )
        return X[:, self.selected_columns]

    def selected_features(self, column_names: Sequence[str]):
        try:
            selected_features = [
                column_names[i] for i in self.selected_columns
            ]
        except NoFeaturesSelectedError as e:
            raise e

        return selected_features


class AnovaSelector(CoreSelector):
    def __init__(self, n_features: int = 10):
        self.n_features = n_features
        self.model = SelectKBest(f_classif, k=self.n_features)
        super().__init__()

    def fit(self, X, y):
        self.model.fit(X, y)
        support = self.model.get_support(indices=True)
        if support is None:
            raise NoFeaturesSelectedError("ANOVA failed to select features.")
        self.selected_columns = support.tolist()


class LassoSelector(CoreSelector):
    def __init__(self):
        self.model = Lasso(random_state=config.SEED)
        super().__init__()

    def optimize_params(self, X, y, verbose=0):
        search = GridSearchCV(
            self.model,
            {"alpha": np.logspace(-5, 1, num=100)},
            cv=5,
            scoring="neg_mean_squared_error",
            verbose=verbose,
        )
        search.fit(X, y)
        best_params = search.best_params_
        log.info(f"Best params for Lasso: {best_params}")
        self.model = self.model.set_params(**best_params)

    def fit(self, X, y):
        self.optimize_params(X, y)
        self.model.fit(X, y)
        coefficients = self.model.coef_
        importance = np.abs(coefficients)
        self.selected_columns = np.where(importance > 0)[0].tolist()
        if not self.selected_columns:
            raise NoFeaturesSelectedError("Lasso failed to select features.")


class BorutaSelector(CoreSelector):
    def fit(self, X, y, verbose=0):
        model = BorutaPy(
            RandomForestClassifier(
                max_depth=5, n_jobs=-1, random_state=config.SEED
            ),
            n_estimators="auto",
            verbose=verbose,
            random_state=config.SEED,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        self.selected_columns = np.where(model.support_)[0].tolist()
        if not self.selected_columns:
            raise NoFeaturesSelectedError("Lasso failed to select features.")


class BorutaSHAPSelector(CoreSelector):
    def fit(self, X, y, verbose=0):
        model = BorutaShap(importance_measure="shap", classification=True)
        # BorutaShap requires X to be pd.DataFrame
        colnames = np.arange(X.shape[1]).astype(str)
        X_df = pd.DataFrame(X, columns=colnames)
        model.fit(
            X=X_df, y=y, n_trials=10, sample=False, verbose=bool(verbose)
        )
        selected_columns_str = model.Subset().columns
        self.selected_columns = [int(c) for c in selected_columns_str]


class FeatureSelectorFactory:
    def __init__(self):
        self.selectors = {
            "anova": AnovaSelector,
            "lasso": LassoSelector,
            "boruta": BorutaSelector,
            "boruta_shap": BorutaSHAPSelector,
        }

    def register_selector(self, name, selector):
        self.selectors[name] = selector

    def get_selector(self, name, *args, **kwargs):
        selector = self.selectors[name]
        if not selector:
            raise ValueError(f"Unknown feature selection ({name}).")
        return selector(*args, **kwargs)


def create_feature_selector(
    method: str = "anova",
    *args,
    **kwargs,
):
    selector = FeatureSelectorFactory().get_selector(method, *args, **kwargs)
    return selector
