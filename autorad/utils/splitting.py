from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.model_selection import (
    LeaveOneOut,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)

from autorad.config import config


def split_cross_validation(
    ids_train: Sequence[str],
    y_train: Sequence[str],
    n_splits: int = 5,
    random_state: Optional[int] = None,
    cv_type: str = "stratified_kfold",
) -> dict[str, dict[str, Any]]:
    """Split the training set into K folds for cross-validation.

    Args:
        ids_train: A Sequence of unique identifiers for the training data points.
        y_train: A Sequence of labels for the training data points.
        n_splits: The number of folds to use for cross-validation.
        random_state: The seed for the random number generator.
        cv_type: The type of cross-validation to use (e.g. "stratified_kfold").

    Returns:
        A Sequence of dictionaries, where each dictionary contains the train and
        validation sets for a single fold of cross-validation, as well as the
        corresponding labels.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    if cv_type == "stratified_kfold":
        kf = StratifiedKFold(
            n_splits=n_splits, random_state=random_state, shuffle=True
        )
    elif cv_type == "repeated_stratified_kfold":
        kf = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=10, random_state=random_state
        )
    elif cv_type == "leave_one_out":
        kf = LeaveOneOut()
    else:
        raise ValueError(f"Unknown cross-validation type: {cv_type}")

    ids_train_cv = {}
    for i, (train_index, val_index) in enumerate(kf.split(ids_train, y_train)):
        ids_train_fold, ids_val_fold = (
            np.asarray(ids_train)[train_index].tolist(),
            np.asarray(ids_train)[val_index].tolist(),
        )
        fold = {"train": ids_train_fold, "val": ids_val_fold}
        ids_train_cv[f"fold_{i}"] = fold
    return ids_train_cv


def split_full_dataset(
    ids: Sequence[str],
    labels: Sequence[str],
    test_size: float = 0.2,
    n_splits: int = 5,
    random_state: Optional[int] = None,
    cv_type: str = "stratified_kfold",
    split_fn: Callable[
        [Sequence[str], Sequence[str], float],
        Tuple[Sequence[str], Sequence[str]],
    ] = train_test_split,
) -> Dict[str, Union[List[str], Dict[str, Any], str]]:
    """
    Split data into test and training, then divide training into n folds for
    cross-validation. Labels are needed for stratification.
    Save the splits as json.
    """
    if test_size < 0 or test_size > 1:
        raise ValueError("test_size must be between 0 and 1")
    result = {
        "split_type": f"{test_size:.0%} test + {n_splits}-fold {cv_type} cross-validation on the training set"
    }
    ids_train, ids_test, labels_train, labels_test = split_fn(
        ids,
        labels,
        stratify=labels,
        test_size=test_size,
        random_state=random_state,
    )
    ids_train_cv = split_cross_validation(
        ids_train,
        labels_train,
        n_splits=n_splits,
        random_state=random_state,
        cv_type=cv_type,
    )
    result.update(
        {
            "test": ids_test,
            "train": ids_train_cv,
        }
    )
    return result


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
    ids_train, ids_val = train_test_split(
        ids_train_val,
        test_size=val_proportion_without_test,
        stratify=y_train_val,
        random_state=random_state,
    )
    train_size = 1 - val_size - test_size
    ids_split = {
        "split_type": f"stratified split: {train_size:.0%} train + {val_size:.0%} validation"
        f" + {test_size:.0%} test",
        "train": ids_train,
        "val": ids_val,
        "test": ids_test,
    }
    return ids_split
