import pandas as pd
import pytest

from autorad.data.dataset import FeatureDataset
from autorad.preprocessing.preprocessor import Preprocessor


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
    "feature_selection_method, oversampling_method, kwargs",
    [
        (None, "SMOTE", {}),
        ("lasso", "SMOTE", {}),
        ("boruta", "BorderlineSMOTE", {}),
        ("anova", "SMOTE", {"n_features": 1}),
        (None, None, {}),
    ],
)
def test_fit_transform(
    feature_dset,
    feature_selection_method,
    oversampling_method,
    kwargs,
):
    preprocessor = Preprocessor(
        feature_selection_method=feature_selection_method,
        oversampling_method=oversampling_method,
        **kwargs,
    )
    data = preprocessor.fit_transform(feature_dset.data)
    assert data._X_preprocessed is not None
    assert isinstance(data._X_preprocessed.train, pd.DataFrame)
    assert data._y_preprocessed is not None
    assert isinstance(data._y_preprocessed.train, pd.Series)

    if feature_selection_method is not None:
        assert data._X_preprocessed.train.columns == ["original_feature_e"]
