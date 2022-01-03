import numpy as np
from sklearn import model_selection
from collections import OrderedDict
from classrad.utils import io
from classrad.config import config


def split_train_test(
    ids, labels, test_size=0.2, shuffle=True, random_state=42
):
    """
    Split data into training and test set.
    :param ids: list of ids
    :param labels: list of labels
    :return: train_ids, test_ids, train_labels, test_labels
    """
    # Split data into training and test set
    ids_train, ids_test, y_train, y_test = model_selection.train_test_split(
        ids,
        labels,
        test_size=test_size,
        stratify=labels,
        shuffle=shuffle,
        random_state=random_state,
    )
    return ids_train, ids_test, y_train, y_test


def split_cross_validation(
    ids, labels, n_splits=5, shuffle=True, random_state=42
):
    """
    Split data into n folds for cross-validation
    """
    ids = np.array(ids)
    ids_split = OrderedDict()
    cv = model_selection.StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )
    split_indices = list(cv.split(ids, labels))
    for i in range(len(split_indices)):
        train_indices, test_indices = list(split_indices[i])
        train_ids = ids[train_indices].tolist()
        test_ids = ids[test_indices].tolist()
        ids_split[f"fold_{i}"] = (train_ids, test_ids)
    return ids_split


def split_full_dataset(
    ids,
    labels,
    result_dir,
    test_size=0.2,
    n_splits=5,
    shuffle=True,
    random_state=config.SEED,
):
    """
    Split data into test and training. Divide training into n folds for
    cross-validation.
    Saves the splits as json.
    """
    ids_train, ids_test, y_train, y_test = split_train_test(
        ids,
        labels,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
    )
    ids_split = {}
    ids_split["test"] = ids_test
    ids_split["train"] = split_cross_validation(
        ids_train, y_train, n_splits=n_splits, random_state=random_state
    )
    save_path = result_dir / "splits.json"
    io.save_json(ids_split, save_path)
    return ids_split
