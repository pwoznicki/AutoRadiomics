import pandas as pd
import pytest

from autorad.data.dataset import FeatureDataset
from autorad.preprocessing.preprocessor import Preprocessor


@pytest.fixture
def feature_dset():
    label = [0, 1] * 25
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
    dset = FeatureDataset(
        dataframe=df,
        target="label",
        ID_colname="id",
    )
    return dset


class TestPreprocessor:
    def setup_method(self):
        self.preprocessor = Preprocessor()

    def test_fit_transform(self, feature_dset):
        feature_dset.split(method="train_val_test")
        data = self.preprocessor.fit_transform(feature_dset.data)
        assert isinstance(data._X_preprocessed, pd.DataFrame)
        assert data._X_preprocessed.columns == ["feature_e"]
