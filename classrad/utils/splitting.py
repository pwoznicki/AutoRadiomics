from typing import List, Union

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from classrad.config import config
from classrad.config.type_definitions import PathLike
from classrad.utils import io


def split_cross_validation(
    ids: List[Union[str, int]],
    labels: List[Union[str, int]],
    n_splits: int = 5,
    random_state: int = config.SEED,
):
    """
    Split data into n_splits folds for cross-validation, with stratification.
    """
    ids_array = np.asarray(ids)
    ids_split = {}
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    for i, (train_indices, test_indices) in enumerate(cv.split(ids, labels)):
        train_ids = ids_array[train_indices].tolist()
        test_ids = ids_array[test_indices].tolist()
        ids_split[f"fold_{i}"] = (train_ids, test_ids)
    return ids_split


def split_full_dataset(
    ids: List[str],
    labels: List[str],
    save_path: PathLike,
    test_size: float = 0.2,
    n_splits: int = 5,
    random_state: int = config.SEED,
):
    """
    Split data into test and training, then divide training into n folds for
    cross-validation. Labels are needed for stratification.
    Save the splits as json.
    """
    ids_train, ids_test, y_train, y_test = train_test_split(
        ids,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    ids_split = {}
    ids_split["test"] = ids_test
    ids_split["train"] = split_cross_validation(
        ids_train, y_train, n_splits=n_splits, random_state=random_state
    )
    io.save_json(ids_split, save_path)
    return ids_split


def split_train_val_test(
    ids: List[str],
    labels: List[str],
    save_path: PathLike,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = config.SEED,
):
    """
    Stratified train/val/test split without cross-validation.
    """
    ids_train_val, ids_test, y_train_val, y_test = train_test_split(
        ids,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    val_proportion_without_test = val_size / (1 - test_size)
    ids_train, ids_val, y_train, y_val = train_test_split(
        ids_train_val,
        y_train_val,
        test_size=val_proportion_without_test,
        stratify=y_train_val,
        random_state=random_state,
    )
    ids_split = {}
    ids_split["train"] = ids_train
    ids_split["val"] = ids_val
    ids_split["test"] = ids_test
    io.save_json(ids_split, save_path)
    return ids_split
