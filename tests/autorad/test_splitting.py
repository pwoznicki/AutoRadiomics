from collections import Counter

import numpy as np

from autorad.utils.splitting import (
    split_cross_validation,
    split_full_dataset,
    split_train_val_test,
)


def test_split_full_dataset(binary_df, multiclass_df):
    for df in [binary_df, multiclass_df]:
        ids = df["id"].tolist()
        labels = df["Label"].tolist()
        splits = split_full_dataset(ids=ids, labels=labels)
        assert len(splits["test"]) == 20
        for i in range(5):
            assert len(splits["train"][f"fold_{i}"]["val"]) == 16


def test_split_cross_validation(binary_df, multiclass_df):
    for df in [binary_df, multiclass_df]:
        ids = df["id"].tolist()
        labels = df["Label"].tolist()
        splits = split_cross_validation(ids, labels)
        for i in range(5):
            fold_split = splits[f"fold_{i}"]
            assert len(fold_split["val"]) == 20
            assert len(fold_split["train"]) == 80


def test_split_train_val_test():
    # create a list of unique identifiers and corresponding labels
    ids = [
        "id1",
        "id2",
        "id3",
        "id4",
        "id5",
        "id6",
        "id7",
        "id8",
        "id9",
        "id10",
    ]
    labels = ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]
    labels = np.asarray(labels)

    # split the data using the function under test
    ids_split = split_train_val_test(ids, labels, test_size=0.2, val_size=0.2)

    # check that the split has the correct format and the correct train/val/test sizes
    assert set(ids_split.keys()) == {"split_type", "train", "val", "test"}
    assert len(ids_split["train"]) == 6
    assert len(ids_split["val"]) == 2
    assert len(ids_split["test"]) == 2

    # convert the lists of strings to lists of indices
    idx_train = [ids.index(id_) for id_ in ids_split["train"]]
    idx_val = [ids.index(id_) for id_ in ids_split["val"]]
    idx_test = [ids.index(id_) for id_ in ids_split["test"]]

    train_labels = np.take(labels, idx_train)
    val_labels = np.take(labels, idx_val)
    test_labels = np.take(labels, idx_test)

    # check that the labels are stratified within each split
    assert Counter(train_labels) == {"a": 3, "b": 3}
    assert Counter(val_labels) == {"a": 1, "b": 1}
    assert Counter(test_labels) == {"a": 1, "b": 1}
