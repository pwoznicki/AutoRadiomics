import pytest

from classrad.utils.splitting import (
    split_cross_validation,
    split_full_dataset,
    split_train_val_test,
)


def test_split_full_dataset(test_dfs, helpers):
    for df in test_dfs:
        ids = df["id"].tolist()
        labels = df["Label"].tolist()
        save_path = helpers.tmp_dir / "splits.json"
        splits = split_full_dataset(
            ids=ids, labels=labels, save_path=save_path
        )
        assert len(splits["test"]) == 20
        for i in range(5):
            fold_split = splits["train"][f"fold_{i}"]
            assert len(fold_split[1]) == 16


@pytest.mark.parametrize("df", ["binary", "multiclass"], indirect=True)
def test_split_cross_validation(df, mode):
    ids = df["id"].tolist()
    labels = df["Label"].tolist()
    splits = split_cross_validation(ids, labels)
    for i in range(5):
        fold_split = splits[f"fold_{i}"]
        assert len(fold_split[1]) == 20


@pytest.mark.parametrize("df", ["binary", "multiclass"], indirect=True)
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
