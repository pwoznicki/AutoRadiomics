from typing import Sequence, Union

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from autorad.config import config


def split_cross_validation(
    ids: Sequence[Union[str, int]],
    labels: Sequence[Union[str, int]],
    n_splits: int = 5,
    random_state: int = config.SEED,
) -> dict:
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
    ids: Sequence[str],
    labels: Sequence[str],
    test_size: float = 0.2,
    n_splits: int = 5,
    random_state: int = config.SEED,
) -> dict:
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
    ids_train_cv = split_cross_validation(
        ids_train, y_train, n_splits, random_state=random_state
    )
    ids_split = {
        "split_type": "test + cross-validation on the training",
        "test": ids_test,
        "train": ids_train_cv,
    }
    return ids_split


def split_train_val_test(
    ids: Sequence[str],
    labels: Sequence[str],
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = config.SEED,
) -> dict:
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
    return ids_split
