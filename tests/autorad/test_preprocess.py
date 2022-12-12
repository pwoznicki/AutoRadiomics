import pandas as pd
import pytest

from autorad.data.dataset import FeatureDataset
from autorad.preprocessing.preprocess import Preprocessor


@pytest.fixture
def feature_dset():
    label = [0, 0, 0, 1, 1] * 10
    df = pd.DataFrame(
        {
            "id": list(range(50)),
            "original_feature_a": [0] * 50,
            "original_feature_b": [0] * 50,
            "original_feature_c": [0] * 50,
            "original_feature_d": [0] * 50,
            "original_feature_e": label,
            "label": label,
        }
    )
    feature_dset = FeatureDataset(
        dataframe=df,
        target="label",
        ID_colname="id",
    )
    feature_dset.split(method="train_val_test")
    return feature_dset


@pytest.mark.parametrize(
    "feature_selection_method, oversampling_method, feature_selection_kwargs",
    [
        (None, "SMOTE", None),
        ("lasso", "SMOTE", None),
        ("boruta", "BorderlineSMOTE", None),
        ("anova", "SMOTE", {"n_features": 1}),
        (None, None, None),
        ("lasso", None, None),
    ],
)
def test_fit_transform(
    feature_dset,
    feature_selection_method,
    oversampling_method,
    feature_selection_kwargs,
):
    preprocessor = Preprocessor(
        feature_selection_method=feature_selection_method,
        oversampling_method=oversampling_method,
        feature_selection_kwargs=feature_selection_kwargs,
    )
    X, y = preprocessor.fit_transform(feature_dset.data.X, feature_dset.data.y)
    assert X is not None
    assert isinstance(X.train, pd.DataFrame)
    assert y is not None
    assert isinstance(y.train, pd.Series)

    if feature_selection_method is not None:
        assert X.train.columns == ["original_feature_e"]
