"""
Create a dataloader class from a dataframe, load selected columns as X and a
column as Y. Add function to split into training, validation and test sets or
stratified split or cross-validation split.
"""
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    StratifiedGroupKFold,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from Radiomics.utils import io

# import monai


class Dataset:
    def __init__(self, dataframe, features, target, task_name="", random_state=11):
        self.df = dataframe
        self.features = features
        self.target = target
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
        self.cv_splits = []
        self.n_splits = None
        self.best_features = None
        self.scaler = MinMaxScaler()
        self.feature_selector = None

    def stratified_split_dataset(self, test_size=0.2, val_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            stratify=self.y,
            random_state=self.random_state,
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train,
            self.y_train,
            test_size=val_size,
            stratify=self.y_train,
            random_state=self.random_state,
        )
        return (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
        )

    def get_cross_validation_folds_from_idx(self):
        if self.cv_splits is None:
            print("No cross-validation splits found!")
        else:
            for train_idx, val_idx in self.cv_splits:
                self.X_train_fold.append(self.X_train.iloc[train_idx])
                self.X_val_fold.append(self.X_train.iloc[val_idx])
                self.y_train_fold.append(self.y_train.iloc[train_idx])
                self.y_val_fold.append(self.y_train.iloc[val_idx])
        return self        

    def cross_validation_split(self, n_splits=5, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            stratify=self.y,
            random_state=self.random_state,
        )
        kf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        self.cv_splits = list(kf.split(self.X_train, self.y_train))
        self.get_cross_validation_folds_from_idx()
        return self

    def cross_validation_split_by_patient(
        self, patient_colname, n_splits=5, test_size=0.2
    ):
        train_inds, test_inds = next(
            StratifiedGroupKFold(
                n_splits=int(np.round(1 / test_size)),
                shuffle=True,
                random_state=self.random_state,
            ).split(self.df, self.y, groups=self.df[patient_colname])
        )
        self.X_train = self.X.iloc[train_inds]
        self.X_test = self.X.iloc[test_inds]
        self.y_train = self.y.iloc[train_inds]
        self.y_test = self.y.iloc[test_inds]
        df_train = self.df.iloc[train_inds]
        kf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        cv_split_generator = kf.split(
            self.X_train, self.y_train, groups=df_train[patient_colname]
        )
        self.cv_splits = list(cv_split_generator)
        self.get_cross_validation_folds_from_idx()
        return self

    def split_train_test_from_column(self, column_name, test_value):
        self.df_test = self.df[self.df[column_name] == test_value]
        self.df_train = self.df[self.df[column_name] != test_value]
        self.X_test = self.df_test[self.features]
        self.y_test = self.df_test[self.target]
        self.X_train = self.df_train[self.features]
        self.y_train = self.df_train[self.target]
        return self.X_train, self.X_test, self.y_train, self.y_test

    def split_dataset_test_from_column(self, column_name, test_value, val_size=0.2):
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = self.split_train_test_from_column(column_name, test_value)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train,
            self.y_train,
            test_size=val_size,
            random_state=self.random_state,
        )
        return (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
        )

    def load_splits_from_json(self, json_path, id_colname):
        splits = io.load_json(json_path)
        test_ids = splits["test"]
        
        test_rows = self.df[id_colname].isin(test_ids)
        train_rows = ~self.df[id_colname].isin(test_ids)

        self.X_test = self.X.loc[test_rows]
        self.y_test = self.y.loc[test_rows]
        self.X_train = self.X.loc[train_rows]
        self.y_train = self.y.loc[train_rows]

        train_ids = splits["train"]
        self.n_splits = len(train_ids)
        for train_fold_ids, val_fold_ids in train_ids.values():
            
            train_fold_rows = self.df[id_colname].isin(train_fold_ids)
            val_fold_rows = self.df[id_colname].isin(val_fold_ids)

            self.X_train_fold.append(self.X.loc[train_fold_rows])
            self.X_val_fold.append(self.X.loc[val_fold_rows])
            self.y_train_fold.append(self.y.loc[train_fold_rows])
            self.y_val_fold.append(self.y.loc[val_fold_rows])
        return self

    def save_splits_to_json(json_path):
        """TBD"""
        pass

    def split_dataset_temporal(self, test_size=0.2, val_size=0.2):
        """TBD"""
        pass

    def cross_validation_split_test_from_column(
        self, column_name, test_value, n_splits=5
    ):
        """
        Splits into train and test according to `column_name`, then creates k-fold CV splitter on the train.
        Args:
            column_name: column to be used for train-test split
            test_value: value in the `column_name` indicating case should be test set
            n_splits: number of CV splits
        Returns:
            self.cv_split_generator: CV splitter
        """
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = self.split_train_test_from_column(column_name, test_value)
        kf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        self.cv_split_generator = kf.split(self.X_train, self.y_train)
        self.cv_splits = list(self.cv_split_generator)
        self.get_cross_validation_folds_from_idx()
        return self

    def standardize_features(self):
        self.X_train[self.X_train.columns] = self.scaler.fit_transform(self.X_train)
        if self.X_val is None:
            print("X_val not set. Leaving out.")
        else:
            self.X_val[self.X_val.columns] = self.scaler.transform(self.X_val)
        self.X_test[self.X_test.columns] = self.scaler.transform(self.X_test)
        self.X[self.X.columns] = self.scaler.transform(self.X)
        return self

    def standardize_features_cross_validation(self):
        self.X_train_fold[self.X_train_fold.columns] = self.scaler.fit_transform(
            self.X_train_fold
        )
        self.X_val_fold[self.X_val_fold.columns] = self.scaler.transform(
            self.X_val_fold
        )

    def inverse_standardize(self, X):
        X[X.columns] = self.scaler.inverse_transform(X)
        return X

    def select_features(self, k=10):
        if self.X_train is None:
            raise ValueError("Split the data into training, validation and test first.")
        else:
            self.feature_selector = SelectKBest(f_classif, k=k)
            self.feature_selector.fit(self.X_train, self.y_train)
            cols = self.feature_selector.get_support(indices=True)
            self.X_train = self.X_train.iloc[:, cols]
            if self.X_val is not None:
                self.X_val = self.X_val.iloc[:, cols]
            self.X_test = self.X_test.iloc[:, cols]
            self.X = self.X.iloc[:, cols]
            self.best_features = self.X.columns
            print(f"Selected features: {self.best_features.values}")

    def select_features_cross_validation(self):
        if self.X_train is None:
            raise ValueError("Split the data into training and test first.")
        else:
            feature_selector = SelectKBest(f_classif, k=10)
            feature_selector.fit(self.X_train_fold, self.y_train_fold)
            cols = feature_selector.get_support(indices=True)
            self.X_train_fold = self.X_train_fold.iloc[:, cols]
            self.X_val_fold = self.X_val_fold.iloc[:, cols]