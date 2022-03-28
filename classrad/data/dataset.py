from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from classrad.config import config
from classrad.config.type_definitions import PathLike
from classrad.feature_selection.feature_selector import FeatureSelector
from classrad.utils import io
from classrad.utils.splitting import split_full_dataset


@dataclass
class TrainingInput:
    train: pd.DataFrame
    test: pd.DataFrame
    val: pd.DataFrame | None = None
    train_folds: list[pd.DataFrame] | None = None
    val_folds: list[pd.DataFrame] | None = None


@dataclass
class TrainingLabels:
    train: pd.Series
    test: pd.Series
    val: pd.Series | None = None
    train_folds: list[pd.Series] | None = None
    val_folds: list[pd.Series] | None = None


@dataclass
class TrainingMeta:
    train: pd.DataFrame
    test: pd.DataFrame
    val: pd.DataFrame | None = None
    train_folds: list[pd.DataFrame] | None = None
    val_folds: list[pd.DataFrame] | None = None


@dataclass
class TrainingData:
    X: TrainingInput
    y: TrainingLabels
    meta: TrainingMeta

    _X_preprocessed: TrainingInput | None = None
    _y_preprocessed: TrainingLabels | None = None

    @property
    def X_preprocessed(self):
        if self._X_preprocessed is None:
            raise ValueError("Preprocessing not performed!")
        return self._X_preprocessed

    @property
    def selected_features(self):
        if self._X_preprocessed is None:
            raise ValueError("Feature selection not performed!")
        return list(self._X_preprocessed.train.columns)

    def normalize_features(self, scaler=MinMaxScaler()):
        """
        Normalize features using scaler from sklearn.
        """
        result = {"train": self.X.train.copy(), "test": self.X.test.copy()}
        all_cols = self.X.train.columns
        result["train"][all_cols] = scaler.fit_transform(self.X.train)
        result["test"][all_cols] = scaler.transform(self.X.test)
        if self.X.val is not None:
            result["val"] = self.X.val.copy()
            result["val"][all_cols] = scaler.transform(self.X.val)
        if self.X.train_folds is not None and self.X.val_folds is not None:
            (
                result["train_folds"],
                result["val_folds"],
            ) = self._normalize_features_cross_validation(scaler)
        self._X_norm = TrainingInput(**result)

    def _normalize_features_cross_validation(
        self, scaler
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        if self.X.train_folds is None or self.X.val_folds is None:
            raise AttributeError("Folds are not set")
        train_folds_norm, val_folds_norm = [], []
        for train_fold, val_fold in zip(self.X.train_folds, self.X.val_folds):
            train_fold_norm = train_fold.copy()
            train_fold_norm[train_fold.columns] = scaler.transform(train_fold)
            train_folds_norm.append(train_fold_norm)
            val_fold_norm = val_fold.copy()
            val_fold_norm[train_fold.columns] = scaler.transform(val_fold)
            val_folds_norm.append(val_fold_norm)
        return train_folds_norm, val_folds_norm

    def select_features(self, method="anova", k=10):
        if self.X_norm is None:
            raise AttributeError(
                "Perform normalization before feature selection!"
            )
        selector = FeatureSelector(method=method, k=k)
        selected_features = selector.fit(self.X_norm.train, self.y.train)

        self._X_selected = self.drop_unselected_features(
            self.X_norm, selected_features
        )

    def drop_unselected_features(
        self, X: TrainingInput, selected_features: list[str]
    ) -> TrainingInput:
        result = {}
        result["train"] = X.train[selected_features]
        result["test"] = X.test[selected_features]
        if X.val is not None:
            result["val"] = X.val[selected_features]
        if X.train_folds is None or X.val_folds is None:
            raise AttributeError("Folds are not set")
        result.update({"train_folds": [], "val_folds": []})
        for train_fold, val_fold in zip(X.train_folds, X.val_folds):
            result["train_folds"].append(train_fold[selected_features])
            result["val_folds"].append(val_fold[selected_features])
        result_X = TrainingInput(**result)
        return result_X

    def balance_classes(self, method="SMOTE"):
        if self.X_norm_selected is None:
            raise AttributeError(
                "Perform feature selection before oversampling!"
            )
        if method == "SMOTE":
            pass


class FeatureDataset:
    """
    Store the data and labels, split into training/test sets, select features
    and show them.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        features: list[str],
        target: str,
        ID_colname: str,
        task_name: str = "",
        meta_columns: list[str] = [],
        random_state: int = config.SEED,
    ):
        self.df = dataframe
        self.features = features
        self.target = target
        self.ID_colname = ID_colname
        self.task_name = task_name
        self.random_state = random_state
        self.X: pd.DataFrame = self.df[self.features]
        self.y: pd.Series = self.df[self.target]
        self.meta_df = self.df[meta_columns + [ID_colname]]
        self._data: TrainingData | None = None
        self.cv_splits: list[tuple[Any, Any]] | None = None
        self.selected_features: list[str] | None = None
        self.result_dir = config.RESULT_DIR

    @property
    def data(self):
        if self._data is None:
            raise AttributeError(
                "Data is not split into training/validation/test. \
                 Split the data or load splits from JSON."
            )
        else:
            return self._data

    def load_splits_from_json(self, json_path: PathLike):
        """
        JSON file should contain the following keys:
            - 'test': list of test IDs
            - 'train': dict with n keys (default n = 5)):
                - 'fold_{0..n-1}': list of training and
                                   list of validation IDs
        It can be created using `full_split()` defined below.
        """
        splits = io.load_json(json_path)

        test_ids = splits["test"]
        test_rows = self.df[self.ID_colname].isin(test_ids)
        train_rows = ~self.df[self.ID_colname].isin(test_ids)

        # Split dataframe rows
        X, y, meta = {}, {}, {}
        X["test"] = self.X.loc[test_rows]
        y["test"] = self.y.loc[test_rows]
        meta["test"] = self.meta_df.loc[test_rows]
        X["train"] = self.X.loc[train_rows]
        y["train"] = self.y.loc[train_rows]
        meta["train"] = self.meta_df.loc[train_rows]

        train_ids = splits["train"]
        n_splits = len(train_ids)
        self.cv_splits = [
            (train_ids[f"fold_{i}"][0], train_ids[f"fold_{i}"][1])
            for i in range(n_splits)
        ]
        X["train_folds"], X["val_folds"] = [], []
        y["train_folds"], y["val_folds"] = [], []
        meta["train_folds"], meta["val_folds"] = [], []
        for train_fold_ids, val_fold_ids in self.cv_splits:

            train_fold_rows = self.df[self.ID_colname].isin(train_fold_ids)
            val_fold_rows = self.df[self.ID_colname].isin(val_fold_ids)

            X["train_folds"].append(self.X[train_fold_rows])
            X["val_folds"].append(self.X[val_fold_rows])
            y["train_folds"].append(self.y[train_fold_rows])
            y["val_folds"].append(self.y[val_fold_rows])
            meta["train_folds"].append(self.meta_df[train_fold_rows])
            meta["val_folds"].append(self.meta_df[val_fold_rows])
        self._data = TrainingData(
            TrainingInput(**X), TrainingLabels(**y), TrainingMeta(**meta)
        )
        return self

    def full_split(self, save_path, test_size: float = 0.2, n_splits: int = 5):
        """
        Split into test and training, split training into 5 folds.
        Save the splits to json.
        """
        ids = self.df[self.ID_colname].tolist()
        labels = self.df[self.target].tolist()
        split_full_dataset(
            ids=ids,
            labels=labels,
            save_path=save_path,
            test_size=test_size,
            n_splits=n_splits,
        )

    def get_train_test_split_from_column(
        self, column_name: str, test_value: str
    ):
        """
        Use if the splits are already in the dataframe.
        """
        X_train = self.X[self.X[column_name] != test_value]
        y_train = self.y[self.y[column_name] != test_value]
        X_test = self.X[self.X[column_name] == test_value]
        y_test = self.y[self.y[column_name] == test_value]

        return X_train, y_train, X_test, y_test

    def split_dataset_test_from_column(
        self, column_name: str, test_value: str, val_size: float = 0.2
    ):
        data = {}
        (
            X_train_and_val,
            y_train_and_val,
            data["X_test"],
            data["y_test"],
        ) = self.get_train_test_split_from_column(column_name, test_value)
        (
            data["X_train"],
            data["X_val"],
            data["y_train"],
            data["y_val"],
        ) = train_test_split(
            X_train_and_val,
            y_train_and_val,
            test_size=val_size,
            random_state=self.random_state,
        )
        self._data = TrainingData(**data)

    def get_splits_cross_validation(self, X, y, n_splits: int):
        """
        Split using stratified k-fold cross-validation.
        """
        kf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        cv_split_generator = kf.split(X, y)
        self.cv_splits = list(cv_split_generator)
        self._add_cross_validation_folds(X, y)
        return self

    def _add_cross_validation_folds(self, X, y):
        X_train_fold, y_train_fold = [], []
        X_val_fold, y_val_fold = [], []
        if self.cv_splits is None:
            raise ValueError("no cross-validation split available!")
        for _, (train_idx, val_idx) in enumerate(self.cv_splits):
            X_train_fold.append(X.iloc[train_idx])
            y_train_fold.append(y.iloc[train_idx])

            X_val_fold.append(X.iloc[val_idx])
            y_val_fold.append(y.iloc[val_idx])

        return self

    def split_cross_validation_test_from_column(
        self, column_name: str, test_value: str, n_splits: int = 5
    ):
        """
        Splits into train and test according to `column_name`,
        then performs stratified k-fold cross validation split
        on the training set.
        """
        data = {}
        (
            data["X_train"],
            data["y_train"],
            data["X_test"],
            data["y_test"],
        ) = self.get_train_test_split_from_column(column_name, test_value)
        data = self.get_splits_cross_validation(
            data["X_train"], data["y_train"], n_splits=n_splits
        )
        return self


class ImageDataset:
    """
    Stores paths to the images, segmentations and labels
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_colname: str,
        mask_colname: str,
        ID_colname: str | None = None,
    ):
        self.df = df
        self.image_colname = self._set_if_in_df(image_colname)
        self.mask_colname = self._set_if_in_df(mask_colname)
        self.ID_colname = self._set_ID_col(ID_colname)

    def _set_if_in_df(self, colname: str):
        if colname not in self.df.columns:
            raise ValueError(
                f"{colname} not found in columns of the dataframe."
            )
        if self.df[colname].isnull().any():
            raise ValueError(f"{colname} contains null values")
        return colname

    def _set_new_IDs(self):
        print("ID not set. Assigning sequential IDs.")
        self.ID_colname = "ID"
        self.df[self.ID_colname] = self.df.index

    def _set_ID_col_from_given(self, id_colname: str):
        if id_colname not in self.df.columns:
            raise ValueError(f"{id_colname} not in columns of dataframe.")
        ids = self.df[id_colname]
        # assert IDs are unique
        if len(ids.unique()) != len(ids):
            raise ValueError("IDs are not unique!")
        self.ID_colname = id_colname

    def _set_ID_col(self, id_colname: str | None = None):
        if id_colname is None:
            self._set_new_IDs()
        else:
            self._set_ID_col_from_given(id_colname)

    def dataframe(self) -> pd.DataFrame:
        return self.df

    def image_paths(self) -> list[str]:
        return self.df[self.image_colname].to_list()

    def mask_paths(self) -> list[str]:
        return self.df[self.mask_colname].to_list()

    def ids(self) -> list[str]:
        if self.ID_colname is None:
            raise AttributeError("ID is not set.")
        return self.df[self.ID_colname].to_list()
