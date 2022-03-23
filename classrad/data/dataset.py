from __future__ import annotations

from dataclasses import dataclass

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
        self.X = self.df[self.features]
        self.y = self.df[self.target]
        self.meta_df = self.df[meta_columns + [ID_colname]]
        self.data: TrainingData | None = None
        self.cv_splits = None
        self.best_features = None
        self.scaler = MinMaxScaler()
        self.result_dir = config.RESULT_DIR

    def load_splits_from_json(self, json_path: PathLike):
        """
        JSON file should contain the following keys:
            - 'test': list of test IDs
            - 'train': dict with n keys (default n = 5)):
                - 'fold_{0..n-1}': list of training and
                                   list of validation IDs
        """
        splits = io.load_json(json_path)
        test_ids = splits["test"]

        test_rows = self.df[self.ID_colname].isin(test_ids)
        train_rows = ~self.df[self.ID_colname].isin(test_ids)

        data = {}
        # Split dataframe rows
        data.X_test = self.X.loc[test_rows]
        data.y_test = self.y.loc[test_rows]
        data.meta_test = self.meta_df.loc[test_rows]
        data.X_train = self.X.loc[train_rows]
        data.y_train = self.y.loc[train_rows]
        data.meta_train = self.meta_df.loc[train_rows]

        train_ids = splits["train"]
        self.n_splits = len(train_ids)
        self.cv_splits = [
            (train_ids[f"fold_{i}"][0], train_ids[f"fold_{i}"][1])
            for i in range(self.n_splits)
        ]
        for train_fold_ids, val_fold_ids in self.cv_splits:

            train_fold_rows = self.df[self.ID_colname].isin(train_fold_ids)
            val_fold_rows = self.df[self.ID_colname].isin(val_fold_ids)

            data.X_train_fold.append(self.X.loc[train_fold_rows])
            data.X_val_fold.append(self.X.loc[val_fold_rows])
            data.y_train_fold.append(self.y.loc[train_fold_rows])
            data.y_val_fold.append(self.y.loc[val_fold_rows])
            data.meta_train_fold.append(self.meta_df.loc[train_fold_rows])
            data.meta_val_fold.append(self.meta_df.loc[val_fold_rows])
        self.data = data
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

    def split_train_test_from_column(self, column_name: str, test_value: str):
        """
        Use if the splits are already in the dataframe.
        """
        self.df_test = self.df[self.df[column_name] == test_value]
        self.df_train = self.df[self.df[column_name] != test_value]
        self.X_test = self.df_test[self.features]
        self.y_test = self.df_test[self.target]
        self.X_train = self.df_train[self.features]
        self.y_train = self.df_train[self.target]
        return self

    def split_dataset_test_from_column(
        self, column_name: str, test_value: str, val_size: float = 0.2
    ):
        self.split_train_test_from_column(column_name, test_value)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train,
            self.y_train,
            test_size=val_size,
            random_state=self.random_state,
        )
        return self

    def _split_train_with_cross_validation(self, n_splits: int):
        """
        Split training with stratified k-fold cross-validation.
        """
        kf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        cv_split_generator = kf.split(self.X_train, self.y_train)
        self.cv_splits = list(cv_split_generator)
        for _, (train_idx, val_idx) in enumerate(self.cv_splits):
            self._add_cross_validation_fold(train_idx, val_idx)
        return self

    def _add_cross_validation_fold(self, train_index, val_index):
        self.X_train_fold.append(self.X_train.iloc[train_index])
        self.y_train_fold.append(self.y_train.iloc[train_index])

        self.X_val_fold.append(self.X_train.iloc[val_index])
        self.y_val_fold.append(self.y_train.iloc[val_index])

        return self

    def split_cross_validation_test_from_column(
        self, column_name: str, test_value: str, n_splits: int = 5
    ):
        """
        Splits into train and test according to `column_name`,
        then performs stratified k-fold cross validation split
        on the training set.
        """
        self.split_train_test_from_column(column_name, test_value)
        self._split_train_with_cross_validation(n_splits=n_splits)
        return self

    def standardize_features(self):
        """
        Normalize features to the range [0, 1] using MinMaxScaler.
        """
        self.X_train.loc[:, :] = self.scaler.fit_transform(self.X_train)
        if self.X_val is None:
            print("X_val not set. Leaving out.")
        else:
            self.X_val.loc[:, :] = self.scaler.transform(self.X_val)
        self.X_test.loc[:, :] = self.scaler.transform(self.X_test)
        self.X.loc[:, :] = self.scaler.transform(self.X)
        return self

    def standardize_features_cross_validation(self):
        for fold_idx in range(len(self.X_train_fold)):
            self.X_train_fold[fold_idx].loc[:, :] = self.scaler.transform(
                self.X_train_fold[fold_idx]
            )
            self.X_val_fold[fold_idx].loc[:, :] = self.scaler.transform(
                self.X_val_fold[fold_idx]
            )

    def inverse_standardize(self, X):
        X[X.columns] = self.scaler.inverse_transform(X)
        return X

    def drop_unselected_features_from_X(self):
        assert self.best_features is not None
        self.X_train = self.X_train[self.best_features]
        self.X_test = self.X_test[self.best_features]
        if self.X_val is not None:
            self.X_val = self.X_val[self.best_features]
        if self.X_train_fold:
            for fold_idx in range(len(self.X_train_fold)):
                self.X_train_fold[fold_idx] = self.X_train_fold[fold_idx][
                    self.best_features
                ]
                self.X_val_fold[fold_idx] = self.X_val_fold[fold_idx][
                    self.best_features
                ]
        self.X = self.X[self.best_features]


class ImageDataset:
    """
    Stores paths to the images, segmentations and labels
    """

    def __init__(self):
        self.df = None
        self.image_colname = None
        self.mask_colname = None
        self.label_colname = None
        self.ID_colname = None

    def _set_df(self, df: pd.DataFrame):
        self.df = df

    def _set_image_col(self, image_colname: str):
        if image_colname not in self.df.columns:
            raise ValueError(f"{image_colname} not in columns of dataframe. ")
        self.image_colname = image_colname

    def _set_mask_col(self, mask_colname: str):
        if mask_colname not in self.df.columns:
            raise ValueError(f"{mask_colname} not in columns of dataframe.")
        self.mask_colname = mask_colname

    def _set_ID_col(self, id_colname: str = None):
        if self.df is None:
            raise ValueError("DataFrame not set!")
        if id_colname is not None:
            if id_colname not in self.df.columns:
                raise ValueError(f"{id_colname} not in columns of dataframe.")
            else:
                ids = self.df[id_colname]
                # assert IDs are unique
                if len(ids.unique()) != len(ids):
                    raise ValueError("IDs are not unique!")
                self.ID_colname = id_colname
        else:
            print("ID not set. Assigning sequential IDs.")
            self.ID_colname = id_colname
            self.df[self.ID_colname] = self.df.index

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        image_colname: str,
        mask_colname: str | None = None,
        id_colname: str = None,
    ):
        dataset = cls()
        dataset._set_df(df),
        dataset._set_image_col(image_colname),
        dataset._set_mask_col(mask_colname),
        dataset._set_ID_col(id_colname)

        return dataset

    def dataframe(self) -> pd.DataFrame:
        return self.df

    def image_paths(self) -> list[str]:
        return self.df[self.image_colname].to_list()

    def mask_paths(self) -> list[str]:
        return self.df[self.mask_colname].to_list()

    def ids(self) -> list[str]:
        return self.df[self.ID_colname].to_list()
