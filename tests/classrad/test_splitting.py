from autorad.utils.splitting import (
    split_cross_validation,
    split_full_dataset,
    split_train_val_test,
)


def test_split_full_dataset(binary_df, multiclass_df, helpers):
    for df in [binary_df, multiclass_df]:
        ids = df["id"].tolist()
        labels = df["Label"].tolist()
        save_path = helpers.tmp_dir() / "splits.json"
        splits = split_full_dataset(
            ids=ids, labels=labels, save_path=save_path
        )
        assert len(splits["test"]) == 20
        for i in range(5):
            fold_split = splits["train"][f"fold_{i}"]
            assert len(fold_split[1]) == 16


def test_split_cross_validation(binary_df, multiclass_df):
    for df in [binary_df, multiclass_df]:
        ids = df["id"].tolist()
        labels = df["Label"].tolist()
        splits = split_cross_validation(ids, labels)
        for i in range(5):
            fold_split = splits[f"fold_{i}"]
            assert len(fold_split[1]) == 20


def test_split_train_val_test(binary_df, multiclass_df, helpers):
    for df in [binary_df, multiclass_df]:
        ids = df["id"].tolist()
        labels = df["Label"].tolist()
        save_path = helpers.tmp_dir() / "splits.json"
        splits = split_train_val_test(
            ids=ids, labels=labels, save_path=save_path
        )
        assert len(splits["test"]) == 20
        assert len(splits["val"]) == 20
        assert len(splits["train"]) == 60
