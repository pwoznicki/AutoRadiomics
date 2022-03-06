import warnings

import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

from classrad.config import config

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


class FeatureSelector:
    def __init__(self):
        self.best_features = None
        self.feature_selector = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, method: str = "anova", k: int = 10
    ):
        if X is None:
            raise ValueError(
                "Split the data into training, (validation) and test first."
            )
        else:
            if method == "anova":
                self.anova_selection(X, y, k=k)
            elif method == "lasso":
                self.lasso_selection(X, y)
            elif method == "boruta":
                self.boruta_selection(X, y)
            else:
                raise ValueError(
                    f"Unknown method for feature selection ({method}). \
                      Choose from 'anova', 'lasso' and 'boruta'."
                )

            print(f"Selected features: {self.best_features}")
        return self.best_features

    def anova_selection(self, X, y, k):
        assert k is not None, "Number of features must be set for anova!"
        self.feature_selector = SelectKBest(f_classif, k=k)
        self.feature_selector.fit(X, y)
        selected_cols = self.feature_selector.get_support(indices=True)
        self.best_features = X.columns[selected_cols].tolist()

    def lasso_selection(self, X, y):
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
        self.best_features = list(np.array(X.columns)[importance > 0])

    def boruta_selection(self, X, y):
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
        self.best_features = X.columns[self.feature_selector.support_].tolist()

    # def select_features_cross_validation(self):
    #     if self.X_train is None:
    #         raise ValueError("Split the data into training and test first.")
    #     else:
    #         feature_selector = SelectKBest(f_classif, k=10)
    #         feature_selector.fit(self.X_train_fold, self.y_train_fold)
    #         cols = feature_selector.get_support(indices=True)
    #         self.X_train_fold = self.X_train_fold.iloc[:, cols]
    #         self.X_val_fold = self.X_val_fold.iloc[:, cols]

    def fit_transform_dataset(self, dataset, method="anova", k=10):
        self.fit(dataset.X_train, dataset.y_train, method=method, k=k)
        dataset.best_features = self.best_features
        dataset.drop_unselected_features_from_X()
        return dataset

    def inverse_transform(self, X):
        return self.feature_selector.inverse_transform(X)
