from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

from classrad.config import config
from classrad.utils import io
from classrad.utils.splitting import split_full_dataset
from classrad.utils.statistics import compare_groups_not_normally_distributed
from classrad.visualization.matplotlib_utils import get_subplots_dimensions


class FeatureDataset:
    """
    Store the data and labels, split into training/test sets, select features
    and show them.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        features: List[str],
        target: str,
        ID_colname: str,
        task_name: str = "",
        meta_columns: List[str] = [],
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
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.X_train_fold = []
        self.X_val_fold = []
        self.y_train_fold = []
        self.y_val_fold = []
        self.meta_df = self.df[meta_columns + [ID_colname]]
        self.meta_train = None
        self.meta_val = None
        self.meta_test = None
        self.meta_train_fold = []
        self.meta_val_fold = []
        self.cv_splits = None
        self.test_ids = None
        self.best_features = None
        self.scaler = MinMaxScaler()
        self.result_dir = config.RESULT_DIR

    def load_splits_from_json(self, json_path: str):
        """
        JSON file should contain the following keys:
            - 'test': list of test IDs
            - 'train': dict with n keys (default n = 5)):
                - 'fold_{0..n-1}': list of training and
                                   list of validation IDs
        """
        splits = io.load_json(json_path)
        self.test_ids = splits["test"]

        test_rows = self.df[self.ID_colname].isin(self.test_ids)
        train_rows = ~self.df[self.ID_colname].isin(self.test_ids)

        # Split dataframe rows
        self.X_test = self.X.loc[test_rows]
        self.y_test = self.y.loc[test_rows]
        self.meta_test = self.meta_df.loc[test_rows]
        self.X_train = self.X.loc[train_rows]
        self.y_train = self.y.loc[train_rows]
        self.meta_train = self.meta_df.loc[train_rows]

        train_ids = splits["train"]
        self.n_splits = len(train_ids)
        self.cv_splits = [
            (train_ids[f"fold_{i}"], train_ids[f"val_{i}"])
            for i in range(self.n_splits)
        ]
        for train_fold_ids, val_fold_ids in self.cv_splits:

            train_fold_rows = self.df[self.ID_colname].isin(train_fold_ids)
            val_fold_rows = self.df[self.ID_colname].isin(val_fold_ids)

            self.X_train_fold.append(self.X.loc[train_fold_rows])
            self.X_val_fold.append(self.X.loc[val_fold_rows])
            self.y_train_fold.append(self.y.loc[train_fold_rows])
            self.y_val_fold.append(self.y.loc[val_fold_rows])
            self.meta_train_fold.append(self.meta_df.loc[train_fold_rows])
            self.meta_val_fold.append(self.meta_df.loc[val_fold_rows])
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

    def boxplot_by_class(
        self, result_dir, neg_label="Negative", pos_label="Positive"
    ):
        """
        Plot the distributions of the selected features by the label class.
        """
        features = self.best_features
        nrows, ncols, figsize = get_subplots_dimensions(len(features))
        fig = make_subplots(rows=nrows, cols=ncols)
        xlabels = [
            pos_label if label == 1 else neg_label for label in self.y_test
        ]
        xlabels = np.array(xlabels)
        # X_test = self.inverse_standardize(self.X_test)
        for i, feature in enumerate(features):
            y = self.X_test[feature]
            _, p_val = compare_groups_not_normally_distributed(
                y[xlabels == neg_label], y[xlabels == pos_label]
            )
            fig.add_trace(
                go.Box(y=y, x=xlabels, name=f"{feature} p={p_val}"),
                row=i // ncols + 1,
                col=i % ncols + 1,
            )
        fig.update_layout(title_text="Selected features:")
        fig.show()
        fig.write_html(Path(result_dir) / "boxplot.html")
        return fig


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
        mask_colname: Optional[str] = None,
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

    def image_paths(self) -> List[str]:
        return self.df[self.image_colname].to_list()

    def mask_paths(self) -> List[str]:
        return self.df[self.mask_colname].to_list()

    def ids(self) -> List[str]:
        return self.df[self.ID_colname].to_list()
