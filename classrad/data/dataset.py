import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from classrad.config import config
from classrad.config.type_definitions import PathLike
from classrad.utils import io, utils
from classrad.utils.splitting import split_full_dataset

log = logging.getLogger(__name__)


@dataclass
class TrainingInput:
    train: pd.DataFrame
    test: pd.DataFrame
    val: Optional[pd.DataFrame] = None
    train_folds: Optional[List[pd.DataFrame]] = None
    val_folds: Optional[List[pd.DataFrame]] = None


@dataclass
class TrainingLabels:
    train: pd.Series
    test: pd.Series
    val: Optional[pd.DataFrame] = None
    train_folds: Optional[List[pd.DataFrame]] = None
    val_folds: Optional[List[pd.DataFrame]] = None


@dataclass
class TrainingMeta:
    train: pd.DataFrame
    test: pd.DataFrame
    val: Optional[pd.DataFrame] = None
    train_folds: Optional[List[pd.DataFrame]] = None
    val_folds: Optional[List[pd.DataFrame]] = None


@dataclass
class TrainingData:
    X: TrainingInput
    y: TrainingLabels
    meta: TrainingMeta

    _X_preprocessed: Optional[TrainingInput] = None
    _y_preprocessed: Optional[TrainingLabels] = None

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


class FeatureDataset:
    """
    Store the extarcted features and labels, split into training/test sets,
    select features and show them.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target: str,
        ID_colname: str,
        features: Optional[List[str]] = None,
        meta_columns: List[str] = [],
        random_state: int = config.SEED,
    ):
        """
        Args:
            dataframe: table with extracted features
            target: name of the label column
            ID_colname: name of column with unique IDs for each case
            features: feature names
            meta_columns: columns to keep that are not features
            random_state: random seed
        Returns:
            None
        """
        self.df = dataframe
        self.target = target
        self.ID_colname = ID_colname
        self.random_state = random_state
        self.features = self._init_features(features)
        self.X: pd.DataFrame = self.df[self.features]
        self.y: pd.Series = self.df[self.target]
        self.meta_df = self.df[meta_columns + [ID_colname]]
        self._data: Optional[TrainingData] = None
        self.cv_splits: Optional[List[tuple[Any, Any]]] = None
        self.selected_features: Optional[List[str]] = None
        self.result_dir = config.RESULT_DIR

    def _init_features(
        self, features: Optional[List[str]] = None
    ) -> List[str]:
        if features is None:
            all_cols = self.df.columns.tolist()
            features = utils.get_pyradiomics_names(all_cols)
        return features

    @property
    def data(self):
        if self._data is None:
            raise AttributeError(
                "Data is not split into training/validation/test. \
                 Split the data or load splits from JSON."
            )
        else:
            return self._data

    def load_splits_from_json(self, json_path: PathLike, split_on=None):
        """
        JSON file should contain the following keys:
            - 'test': list of test IDs
            - 'train': dict with n keys (default n = 5)):
                - 'fold_{0..n-1}': list of training and
                                   list of validation IDs
        It can be created using `full_split()` defined below.
        """
        splits = io.load_json(json_path)
        if split_on is None:
            split_on = self.ID_colname
        test_ids = splits["test"]
        test_rows = self.df[split_on].isin(test_ids)
        train_rows = ~self.df[split_on].isin(test_ids)

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

            train_fold_rows = self.df[split_on].isin(train_fold_ids)
            val_fold_rows = self.df[split_on].isin(val_fold_ids)

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

    def full_split(
        self,
        save_path: PathLike,
        split_on: Optional[str] = None,
        test_size: float = 0.2,
        n_splits: int = 5,
    ):
        """
        Split into test and training, split training into 5 folds.
        Save the splits to json.
        """
        if split_on is None:
            split_on = self.ID_colname
        patient_df = self.df[[split_on, self.target]].drop_duplicates()
        if not patient_df[split_on].is_unique:
            raise ValueError(
                f"Selected column {split_on} has varying labels for the same ID!"
            )
        ids = patient_df[split_on].tolist()
        labels = patient_df[self.target].tolist()
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

    def split_cross_validation(self, X, y, n_splits: int):
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
        data = self.split_cross_validation(
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
        ID_colname: Optional[str] = None,
    ):
        """
        Args:
            df: dataframe with image and mask paths
            image_colname: name of the image column in df
            mask_colname: name of the mask column in df
            ID_colname: name of the ID column in df, if not given,
                IDs are assigned sequentially
        Returns:
            None
        """
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
        log.info("ID not set. Assigning sequential IDs.")
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

    def _set_ID_col(self, id_colname: Optional[str] = None):
        if id_colname is None:
            self._set_new_IDs()
        else:
            self._set_ID_col_from_given(id_colname)

    def dataframe(self) -> pd.DataFrame:
        return self.df

    def image_paths(self) -> List[str]:
        return self.df[self.image_colname].to_list()

    def mask_paths(self) -> List[str]:
        return self.df[self.mask_colname].to_list()

    def ids(self) -> List[str]:
        if self.ID_colname is None:
            raise AttributeError("ID is not set.")
        return self.df[self.ID_colname].to_list()
