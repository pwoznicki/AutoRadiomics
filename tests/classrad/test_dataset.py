from hypothesis import given, settings
from classrad.data.dataset import Dataset
import hypothesis_utils


class TestDataset:
    @given(df=hypothesis_utils.simple_df())
    @settings(max_examples=5)
    def test_init(self, df):
        dataset = Dataset(
            dataframe=df,
            features=["Feature1"],
            target="Label",
            ID_colname="ID",
            task_name="Testing",
        )
        assert dataset.X.columns == ["Feature1"]
        assert dataset.y.name == "Label"

    @given(df=hypothesis_utils.simple_df())
    @settings(max_examples=5)
    def test_full_split(self, df):
        dataset = Dataset(
            dataframe=df,
            features=["Feature1"],
            target="Label",
            ID_colname="ID",
            task_name="Testing",
        )
        dataset.full_split()

    @given(df=hypothesis_utils.simple_df())
    @settings(max_examples=5)
    def test_cross_validation_split(self, df):
        dataset = Dataset(
            dataframe=df,
            features=["Feature1"],
            target="Label",
            ID_colname="ID",
            task_name="Testing",
        )
        dataset.cross_validation_split()
        assert dataset.X_train.shape[0] == 80
        assert dataset.X_test.shape[0] == 20
        assert dataset.y_train.shape[0] == 80
        assert dataset.y_test.shape[0] == 20
