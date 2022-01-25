import pytest
from classrad.utils.splitting import (
    split_full_dataset,
    split_cross_validation,
    split_train_val_test,
)


@pytest.mark.parametrize("mode", ["binary", "multiclass"])
def test_split_full_dataset(test_df, mode, tmp_path):
    df = test_df[mode]
    ids = df["id"].tolist()
    labels = df["Label"].tolist()
    result_dir = tmp_path / "splits"
    result_dir.mkdir()
    splits = split_full_dataset(ids, labels, result_dir)
    assert len(splits["test"]) == 20
    for i in range(5):
        fold_split = splits["train"][f"fold_{i}"]
        assert len(fold_split[1]) == 16


@pytest.mark.parametrize("mode", ["binary", "multiclass"])
def test_split_cross_validation(test_df, mode):
    df = test_df[mode]
    ids = df["id"].tolist()
    labels = df["Label"].tolist()
    splits = split_cross_validation(ids, labels)
    for i in range(5):
        fold_split = splits[f"fold_{i}"]
        assert len(fold_split[1]) == 20


@pytest.mark.parametrize("mode", ["binary", "multiclass"])
def test_split_train_val_test(test_df, mode, tmp_path):
    df = test_df[mode]
    ids = df["id"].tolist()
    labels = df["Label"].tolist()
    result_dir = tmp_path / "splits"
    result_dir.mkdir()
    splits = split_train_val_test(ids, labels, tmp_path)
    assert len(splits["test"]) == 20
    assert len(splits["val"]) == 20
    assert len(splits["train"]) == 60
