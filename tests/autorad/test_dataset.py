from pathlib import Path

import hypothesis_utils
from hypothesis import given, settings

from autorad.config import config
from autorad.data.dataset import FeatureDataset


class TestFeatureDataset:
    @given(df=hypothesis_utils.simple_df())
    @settings(max_examples=5)
    def test_init(self, df):
        dataset = FeatureDataset(
            dataframe=df,
            features=["Feature1"],
            target="Label",
            ID_colname="ID",
        )
        assert dataset.X.columns == ["Feature1"]
        assert dataset.y.name == "Label"

    test_data_path = Path(config.TEST_DATA_DIR) / "splits.json"
