from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from classrad.config import config
from classrad.config.type_definitions import PathLike
from classrad.utils import io
from classrad.utils.splitting import split_full_dataset


@dataclass
class TrainingData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    meta_train: pd.DataFrame
    meta_test: pd.DataFrame
    X_val: pd.DataFrame | None = None
    y_val: pd.Series | None = None
    meta_val: pd.DataFrame | None = None
    X_train_fold: list[pd.DataFrame] | None = None
    X_val_fold: list[pd.DataFrame] | None = None
    y_train_fold: list[pd.Series] | None = None
    y_val_fold: list[pd.Series] | None = None
    meta_train_fold: list[pd.DataFrame] | None = None
    meta_val_fold: list[pd.DataFrame] | None = None

    X_train_norm: pd.DataFrame | None = None
    X_test_norm: pd.DataFrame | None = None
    X_val_norm: pd.DataFrame | None = None
    X_train_fold_norm: list[pd.DataFrame] | None = None
    X_val_fold_norm: list[pd.DataFrame] | None = None

    def normalize_features(self, scaler=MinMaxScaler()):
        """
        Normalize features to the range [0, 1] using MinMaxScaler.
        """
        self.X_train_norm = scaler.fit_transform(self.X_train)
        self.X_test_norm = scaler.transform(self.X_test)
        if self.X_val is not None:
            self.X_val_norm = scaler.transform(self.X_val)
        if self.X_train_fold is not None and self.X_val_fold is not None:
            self._normalize_features_cross_validation(scaler)
        # self.X.loc[:, :] = self.scaler.transform(self.X)

    def _normalize_features_cross_validation(self, scaler):
        if self.X_train_fold is None or self.X_val_fold is None:
            raise AttributeError("Folds are not set")
        self.X_train_fold_norm, self.X_val_fold_norm = [], []
        for train_fold, val_fold in zip(self.X_train_fold, self.X_val_fold):
            train_fold_norm = scaler.transform(train_fold)
            self.X_train_fold_norm.append(train_fold_norm)
            val_fold_norm = scaler.transform(val_fold)
            self.X_val_fold_norm.append(val_fold_norm)


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
        self.best_features: list[str] | None = None
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
        data = {}
        data["X_test"] = self.X.loc[test_rows]
        data["y_test"] = self.y.loc[test_rows]
        data["meta_test"] = self.meta_df.loc[test_rows]
        data["X_train"] = self.X.loc[train_rows]
        data["y_train"] = self.y.loc[train_rows]
        data["meta_train"] = self.meta_df.loc[train_rows]

        train_ids = splits["train"]
        n_splits = len(train_ids)
        self.cv_splits = [
            (train_ids[f"fold_{i}"][0], train_ids[f"fold_{i}"][1])
            for i in range(n_splits)
        ]
        data.update(
            data.fromkeys(
                [
                    "X_train_fold",
                    "X_val_fold",
                    "y_train_fold",
                    "y_val_fold",
                    "meta_train_fold",
                    "meta_val_fold",
                ],
                [],
            )
        )
        for train_fold_ids, val_fold_ids in self.cv_splits:

            train_fold_rows = self.df[self.ID_colname].isin(train_fold_ids)
            val_fold_rows = self.df[self.ID_colname].isin(val_fold_ids)

            data["X_train_fold"].append(self.X.loc[train_fold_rows])
            data["X_val_fold"].append(self.X.loc[val_fold_rows])
            data["y_train_fold"].append(self.y.loc[train_fold_rows])
            data["y_val_fold"].append(self.y.loc[val_fold_rows])
            data["meta_train_fold"].append(self.meta_df.loc[train_fold_rows])
            data["meta_val_fold"].append(self.meta_df.loc[val_fold_rows])
        self._data = TrainingData(**data)
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

    def drop_unselected_features_from_X(self):
        if self.best_features is None:
            raise AttributeError("Best features not set!")
        self.data.X_train = self.data.X_train[self.best_features]
        self.data.X_test = self.data.X_test[self.best_features]
        if self.data.X_val is not None:
            self.data.X_val = self.data.X_val[self.best_features]
        if (
            self.data.X_train_fold is not None
            and self.data.X_val_fold is not None
        ):
            for fold_idx in range(len(self.data.X_train_fold)):
                self.data.X_train_fold[fold_idx] = self.data.X_train_fold[
                    fold_idx
                ][self.best_features]
                self.data.X_val_fold[fold_idx] = self.data.X_val_fold[
                    fold_idx
                ][self.best_features]
        self.X = self.X[self.best_features]


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
