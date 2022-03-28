from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

from classrad.config import config


class FeatureSelector:
    def __init__(self, method: str = "anova", n_features: int = 10):
        self.method = method
        self.n_features = n_features
        self.selected_features: list[str] | None = None

    def fit(self, X, y) -> list[str]:
        if X is None:
            raise ValueError(
                "Split the data into training, (validation) and test first."
            )
        if self.method == "anova":
            self.selected_features = self.fit_anova(X, y, k=self.n_features)
        elif self.method == "lasso":
            self.selected_features = self.fit_lasso(X, y)
        elif self.method == "boruta":
            self.selected_features = self.fit_boruta(X, y)
        else:
            raise ValueError(
                f"Unknown method for feature selection ({self.method}). \
                    Choose from `anova`, `lasso` and `boruta`."
            )
        print(f"Selected features: {self.selected_features}")
        return self.selected_features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features is None:
            raise ValueError(
                "Call fit() first to select features before transforming."
            )
        return X[self.selected_features]

    def fit_anova(self, X, y, k: int) -> list[str]:
        if k is None:
            raise ValueError("Number of features must be set for anova!")
        self.feature_selector = SelectKBest(f_classif, k=k)
        self.feature_selector.fit(X, y)
        selected_cols = self.feature_selector.get_support(indices=True)
        selected_features = X.columns[selected_cols].tolist()
        return selected_features

    def fit_lasso(self, X, y, cv_splits=None) -> list[str]:
        self.feature_selector = Lasso(random_state=config.SEED)
        if cv_splits is None:
            cv_splits = 5
        search = GridSearchCV(
            self.feature_selector,
            {"alpha": np.arange(0.01, 0.5, 0.005)},
            cv=cv_splits,
            scoring="neg_mean_squared_error",
            verbose=3,
        )
        search.fit(X, y)
        coefficients = search.best_estimator_.coef_
        importance = np.abs(coefficients)
        selected_features = list(np.array(X.columns)[importance > 0])
        return selected_features

    def fit_boruta(self, X, y) -> list[str]:
        self.feature_selector = BorutaPy(
            RandomForestClassifier(
                max_depth=5, n_jobs=-1, random_state=config.SEED
            ),
            n_estimators="auto",
            verbose=2,
            random_state=config.SEED,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.feature_selector.fit(X.values, y.values)
        selected_features = X.columns[self.feature_selector.support_].tolist()
        return selected_features
