from __future__ import annotations

import warnings

import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

# from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

from classrad.config import config

# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=ConvergenceWarning)


class FeatureSelector:
    def __init__(self):
        self.feature_selector = None

    def fit(self, X, y, method: str = "anova", k: int = 10) -> list[str]:
        if X is None:
            raise ValueError(
                "Split the data into training, (validation) and test first."
            )
        if method == "anova":
            selected_features = self.anova_selection(X, y, k=k)
        elif method == "lasso":
            selected_features = self.lasso_selection(X, y)
        elif method == "boruta":
            selected_features = self.boruta_selection(X, y)
        else:
            raise ValueError(
                f"Unknown method for feature selection ({method}). \
                    Choose from `anova`, `lasso` and `boruta`."
            )
        print(f"Selected features: {selected_features}")
        return selected_features

    def anova_selection(self, X, y, k: int) -> list[str]:
        if k is None:
            raise ValueError("Number of features must be set for anova!")
        self.feature_selector = SelectKBest(f_classif, k=k)
        self.feature_selector.fit(X, y)
        selected_cols = self.feature_selector.get_support(indices=True)
        selected_features = X.columns[selected_cols].tolist()
        return selected_features

    def lasso_selection(self, X, y) -> list[str]:
        self.feature_selector = Lasso(random_state=config.SEED)
        search = GridSearchCV(
            self.feature_selector,
            {"alpha": np.arange(0.01, 0.5, 0.005)},
            cv=5,
            scoring="neg_mean_squared_error",
            verbose=3,
        )
        search.fit(X, y)
        coefficients = search.best_estimator_.coef_
        importance = np.abs(coefficients)
        selected_features = list(np.array(X.columns)[importance > 0])
        return selected_features

    def boruta_selection(self, X, y) -> list[str]:
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
