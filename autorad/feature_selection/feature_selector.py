from __future__ import annotations

import abc
import warnings
from typing import Sequence

import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

from autorad.config import config


class NoFeaturesSelectedError(Exception):
    """raised when feature selection fails"""

    pass


class CoreSelector(abc.ABC):
    """Template for feature selection methods"""

    def __init__(self):
        self.selected_columns = None

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> list[int]:
        pass

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return X[:, self.selected_columns]

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_columns is None:
            raise ValueError(
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
        super().__init__()

    def fit(self, X, y):
        model = SelectKBest(f_classif, k=self.n_features)
        model.fit(X, y)
        support = model.get_support(indices=True)
        if support is None:
            raise ValueError("ANOVA failed to select features.")
        self.selected_columns = support.tolist()


class LassoSelector(CoreSelector):
    def __init__(self):
        super().__init__()

    def fit(self, X, y, verbose=3):
        model = Lasso(random_state=config.SEED)
        search = GridSearchCV(
            model,
            {"alpha": np.arange(0.1, 10, 0.1)},
            cv=5,
            scoring="neg_mean_squared_error",
            verbose=verbose,
        )
        search.fit(X, y)
        coefficients = search.best_estimator_.coef_
        importance = np.abs(coefficients)
        self.selected_columns = np.where(importance > 0)[0].tolist()
        if not self.selected_columns:
            raise ValueError("Lasso failed to select features.")


class BorutaSelector(CoreSelector):
    def fit(self, X, y, verbose=3):
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
            raise ValueError("Boruta failed to select features.")


class FeatureSelectorFactory:
    def __init__(self):
        self.selectors = {
            "anova": AnovaSelector,
            "lasso": LassoSelector,
            "boruta": BorutaSelector,
        }

    def register_selector(self, name, selector):
        self.selectors[name] = selector

    def get_selector(self, name, *args, **kwargs):
        selector = self.selectors[name]
        if not selector:
            raise ValueError(f"Unknown feature selection ({name}).")
        return selector(*args, **kwargs)


def create_feature_selector(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "anova",
    n_features: int | None = None,
):
    selector = FeatureSelectorFactory().get_selector(method, n_features)
    return selector
