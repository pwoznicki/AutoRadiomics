import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from autorad.config import config
from autorad.config.type_definitions import PathLike
from autorad.utils import extraction_utils, io, splitting
from autorad.visualization import matplotlib_utils, plot_volumes

log = logging.getLogger(__name__)


@dataclass
class TrainingInput:
    train: pd.DataFrame
    test: pd.DataFrame
    val: Optional[pd.DataFrame] = None
    train_folds: Optional[list[pd.DataFrame]] = None
    val_folds: Optional[list[pd.DataFrame]] = None

    def __post_init__(self):
        assert (self.val is not None and self.train_folds is None) or (
            self.val is None and self.train_folds is not None
        ), "Exactly one of val, train_folds should be not None"


@dataclass
class TrainingLabels:
    train: pd.Series
    test: pd.Series
    val: Optional[pd.DataFrame] = None
    train_folds: Optional[list[pd.DataFrame]] = None
    val_folds: Optional[list[pd.DataFrame]] = None

    def __post_init__(self):
        assert (self.val is not None and self.train_folds is None) or (
            self.val is None and self.train_folds is not None
        ), "Exactly one of val, train_folds should be not None"


@dataclass
class TrainingMeta:
    train: pd.DataFrame
    test: pd.DataFrame
    val: Optional[pd.DataFrame] = None
    train_folds: Optional[list[pd.DataFrame]] = None
    val_folds: Optional[list[pd.DataFrame]] = None


@dataclass
class TrainingData:
    X: TrainingInput
    y: TrainingLabels
    meta: TrainingMeta

    def iter_training(self):
        if self.X.val is not None:
            yield self.X.train, self.y.train, self.meta.train, self.X.val, self.y.val, self.meta.val
        elif self.X.train_folds is not None:
            yield from zip(
                self.X.train_folds,
                self.y.train_folds,
                self.meta.train_folds,
                self.X.val_folds,
                self.y.val_folds,
                self.meta.val_folds,
            )

    def __repr__(self):
        return (
            f"TrainingData with {len(self.y.train)} training observations, "
            f"{len(self.y.test)} test observations, {self.X.train.shape[1]} feature columns "
            f"and {self.meta.train.shape[1]} meta columns."
        )


class FeatureDataset:
    """
    Store and manage extracted features and labels from a dataset.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target: str,
        ID_colname: str,
        features: Optional[list[str]] = None,
        meta_columns: list[str] = [],
        random_state: int = config.SEED,
    ):
        """
        Args:
            dataframe: table with extracted features
            target: name of the label column
            ID_colname: name of column with unique IDs for each case
            features: feature names
            meta_columns: columns to keep that are not features
            random_state: random seed for splitting data
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
        self.cv_splits: Optional[list[tuple[Any, Any]]] = None

    def _init_features(
        self, features: Optional[list[str]] = None
    ) -> list[str]:
        if features is None:
            all_cols = self.df.columns.tolist()
            features = extraction_utils.filter_pyradiomics_names(all_cols)
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

    @property
    def splits(self):
        if self._splits is None:
            raise AttributeError("No splits loaded. Split the data first.")
        return self._splits

    def load_splits_from_json(self, json_path: PathLike):
        splits = io.load_json(json_path)
        return self.load_splits(splits)

    def load_splits(self, splits: dict):
        """
        Load training and test splits from a dictionary.

        `splits` dictionary should contain the following keys:
            - 'test': list of test IDs
            - 'train': dict with n keys (default n = 5)):
                - 'fold_{0..n-1}': list of training and
                                   list of validation IDs
            - 'split_on'; column name to split on,
                if None, split is performed on ID_colname
        It can be created using `full_split()` defined below.
        """
        self._splits = splits
        split_on = splits["split_on"]

        test_ids = splits["test"]
        test_rows = self.df[split_on].isin(test_ids)

        # Split dataframe rows
        X, y, meta = {}, {}, {}

        # Split the test data
        X["test"] = self.X.loc[test_rows]
        y["test"] = self.y.loc[test_rows]
        meta["test"] = self.meta_df.loc[test_rows]

        train_rows = ~self.df[split_on].isin(test_ids)

        # Split the validation data, if specified
        if "val" in splits:
            val_ids = splits["val"]
            val_rows = self.df[split_on].isin(val_ids)
            train_rows = train_rows & ~val_rows

            X["val"] = self.X.loc[val_rows]
            y["val"] = self.y.loc[val_rows]
            meta["val"] = self.meta_df.loc[val_rows]

        # Split the training data
        X["train"] = self.X.loc[train_rows]
        y["train"] = self.y.loc[train_rows]
        meta["train"] = self.meta_df.loc[train_rows]

        if isinstance(splits["train"], dict):
            train_splits = splits["train"]
            n_splits = len(train_splits)
            self.cv_splits = [
                train_splits[f"fold_{i}"] for i in range(n_splits)
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

    def split(
        self,
        save_path: Optional[PathLike] = None,
        method="train_with_cross_validation_test",
        split_on: Optional[str] = None,
        test_size: float = 0.2,
        *args,
        **kwargs,
    ) -> dict:
        if split_on is None:
            split_on = self.ID_colname
        if method == "train_with_cross_validation_test":
            splits = self.full_split(split_on, test_size, *args, **kwargs)
        elif method == "train_val_test":
            splits = self.split_train_val_test(
                split_on, test_size, *args, **kwargs
            )
        else:
            raise ValueError(f"Method {method} is not supported.")

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            io.save_yaml(splits, save_path)
        self.load_splits(splits)
        return splits

    def full_split(
        self,
        split_on: str,
        test_size: float = 0.2,
        n_splits: int = 5,
    ) -> dict:
        """
        Split into test and training, split training into k folds.
        """
        patient_df = self.df[[split_on, self.target]].drop_duplicates()
        if not patient_df[split_on].is_unique:
            raise ValueError(
                f"Selected column {split_on} has varying labels for the same ID!"
            )
        ids = patient_df[split_on].tolist()
        labels = patient_df[self.target].tolist()
        splits = splitting.split_full_dataset(
            ids=ids,
            labels=labels,
            test_size=test_size,
            n_splits=n_splits,
        )
        splits["split_on"] = split_on
        return splits

    def split_train_val_test(
        self,
        split_on: str,
        test_size: float = 0.2,
        val_size: float = 0.2,
    ):
        ids = self.df[split_on].tolist()
        labels = self.df[self.target].tolist()
        splits = splitting.split_train_val_test(
            ids=ids,
            labels=labels,
            test_size=test_size,
            val_size=val_size,
        )
        splits["split_on"] = split_on
        return splits

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

    def split_train_val_test_from_column(
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

    def full_split_with_test_from_column(
        self,
        column_name: str,
        test_value: str,
        save_path: PathLike,
        split_on: Optional[str] = None,
        split_label: Optional[str] = None,
        n_splits: int = 5,
    ):
        """
        Splits into train and test according to `column_name`,
        then performs stratified k-fold cross validation split
        on the training set.
        """
        if split_on is None:
            split_on = self.ID_colname
        if split_label is None:
            split_label = self.target
        df_to_split = self.df[
            [split_on, split_label, column_name]
        ].drop_duplicates()
        if not df_to_split[split_on].is_unique:
            raise ValueError(
                f"Selected column {split_on} has varying labels for the same ID!"
            )
        train_to_split = df_to_split[df_to_split[column_name] != test_value]
        ids_train = train_to_split[split_on].tolist()
        y_train = train_to_split[split_label].tolist()
        ids_test = df_to_split.loc[
            df_to_split[column_name] == test_value, split_on
        ].tolist()

        ids_train_cv = splitting.split_cross_validation(
            ids_train, y_train, n_splits, random_state=self.random_state
        )

        ids_split = {
            "split_type": f"predefined test as {column_name} = {test_value}"
            " and stratified cross validation on training",
            "split_on": split_on,
            "test": ids_test,
            "train": ids_train_cv,
        }
        io.save_json(ids_split, save_path)
        self.load_splits(ids_split)

        return self


class ImageDataset:
    """
    Stores paths to the images, segmentations and labels
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_colname: str = "image_path",
        mask_colname: str = "segmentation_path",
        ID_colname: Optional[str] = None,
        root_dir: Optional[PathLike] = None,
    ):
        """
        Args:
            df: dataframe with image and mask paths
            image_colname: name of the image column in df
            mask_colname: name of the mask column in df
            ID_colname: name of the ID column in df. If None,
                IDs are assigned sequentially
            root_dir: root directory of the dataset, if needed
                to resolve paths
        """
        self._df = df
        self.image_colname = self._check_if_in_df(image_colname)
        self.mask_colname = self._check_if_in_df(mask_colname)
        self._set_ID_col(ID_colname)
        self.root_dir = root_dir

    def _check_if_in_df(self, colname: str):
        if colname not in self._df.columns:
            raise ValueError(
                f"{colname} not found in columns of the dataframe."
            )
        if self._df[colname].isnull().any():
            raise ValueError(f"{colname} contains null values")
        return colname

    def _set_new_IDs(self):
        log.info("ID not set. Assigning sequential IDs.")
        if "ID" not in self._df.columns:
            self.ID_colname = "ID"
        else:
            self.ID_colname = "ID_autogenerated"
        self._df.insert(0, self.ID_colname, range(len(self._df)))

    def _set_ID_col_from_given(self, id_colname: str):
        if id_colname not in self._df.columns:
            raise ValueError(f"{id_colname} not in columns of dataframe.")
        ids = self._df[id_colname]
        # assert IDs are unique
        if len(ids.unique()) != len(ids):
            raise ValueError("IDs are not unique!")
        self.ID_colname = id_colname

    def _set_ID_col(self, id_colname: Optional[str] = None):
        if id_colname is None:
            self._set_new_IDs()
        else:
            self._set_ID_col_from_given(id_colname)

    @property
    def df(self) -> pd.DataFrame:
        """If root_dir is set, returns the dataframe with paths resolved"""
        if self.root_dir is None:
            return self._df
        result = self._df.copy()
        return result.assign(
            **{
                self.image_colname: self._df[self.image_colname].apply(
                    lambda x: os.path.join(self.root_dir, x)
                )
            }
        ).assign(
            **{
                self.mask_colname: self._df[self.mask_colname].apply(
                    lambda x: os.path.join(self.root_dir, x)
                )
            }
        )

    @property
    def image_paths(self) -> list[str]:
        return self.df[self.image_colname].to_list()

    @property
    def mask_paths(self) -> list[str]:
        return self.df[self.mask_colname].to_list()

    @property
    def ids(self) -> list[str]:
        return self.df[self.ID_colname].to_list()

    def plot_examples(self, n: int = 1, window="soft tissues"):
        if n > len(self.image_paths):
            n = len(self.image_paths)
            log.info(
                f"Not enough cases. Plotting all the cases instead (n={n})"
            )
        df_to_plot = self.df.sample(n)
        nrows, ncols, figsize = matplotlib_utils.get_subplots_dimensions(n)
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        for i in range(len(df_to_plot)):
            case = df_to_plot.iloc[i]
            ax = axs.flat[i]
            vols = plot_volumes.BaseVolumes.from_nifti(
                case[self.image_colname],
                case[self.mask_colname],
                window=window,
            )
            image_2D, mask_2D = vols.get_slices()
            single_plot = plot_volumes.overlay_mask_contour(image_2D, mask_2D)
            ax.imshow(single_plot)
            ax.set_title(f"{case[self.ID_colname]}")
            ax.axis("off")
        # return fig
