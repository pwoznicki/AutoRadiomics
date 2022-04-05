from __future__ import annotations

import warnings

import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

from classrad.config import config


class FeatureSelector:
    def __init__(
        self, method: str = "anova", n_features: int = 10, test_cols=None
    ):
        self.method = method
        self.n_features = n_features
        self.test_cols = test_cols
        self.selected_columns: list[int] | None = None
        self.selected_features: list[str] | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        column_names: list[str] | None = None,
    ) -> None:
        if self.method == "anova":
            self.selected_columns = self.fit_anova(X, y, k=self.n_features)
        elif self.method == "lasso":
            self.selected_columns = self.fit_lasso(X, y)
        elif self.method == "boruta":
            self.selected_columns = self.fit_boruta(X, y)
        else:
            raise ValueError(
                f"Unknown method for feature selection ({self.method}). \
                    Choose from `anova`, `lasso` and `boruta`."
            )
        if column_names is not None:
            self.selected_features = [
                column_names[i] for i in self.selected_columns
            ]

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        column_names: list[str] | None = None,
    ) -> list[int]:
        self.fit(X, y, column_names)

        return X[:, self.selected_columns], y

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_columns is None:
            raise ValueError(
                "Call fit() first to select features before transforming."
            )
        return X[:, self.selected_columns]

    def fit_anova(self, X: np.ndarray, y: np.ndarray, k: int) -> list[int]:
        if k is None:
            raise ValueError("Number of features must be set for anova!")
        model = SelectKBest(f_classif, k=k)
        model.fit(X, y)
        support = model.get_support(indices=True)
        assert support is not None, "ANOVA failed to select features."
        selected_columns = support.tolist()
        return selected_columns

    def fit_lasso(
        self, X: np.ndarray, y: np.ndarray, cv_splits=None
    ) -> list[int]:
        model = Lasso(random_state=config.SEED)
        if cv_splits is None:
            cv_splits = 5
        search = GridSearchCV(
            model,
            {"alpha": np.arange(0.01, 0.5, 0.005)},
            cv=cv_splits,
            scoring="neg_mean_squared_error",
            verbose=3,
        )
        search.fit(X, y)
        coefficients = search.best_estimator_.coef_
        importance = np.abs(coefficients)
        selected_columns = np.where(importance > 0.01)[0].tolist()
        assert selected_columns, "Lasso failed to select features."
        return selected_columns

    def fit_boruta(self, X: np.ndarray, y: np.ndarray) -> list[int]:
        model = BorutaPy(
            RandomForestClassifier(
                max_depth=5, n_jobs=-1, random_state=config.SEED
            ),
            n_estimators="auto",
            verbose=2,
            random_state=config.SEED,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        selected_columns = np.where(model.support_)[0].tolist()
        assert selected_columns, "Boruta failed to select features."
        return selected_columns
