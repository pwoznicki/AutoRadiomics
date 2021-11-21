from hypothesis.extra.pandas import data_frames, column, range_indexes
from hypothesis import given, settings, strategies as st
from sklearn import feature_selection
from classrad.data.dataset import Dataset

simple_df = data_frames(
    columns=[
        column(name="ID", elements=st.integers()),
        column(name="Label", elements=st.booleans()),
        column(name="Feature1", elements=st.floats(min_value=0.0, max_value=1.0)),
        column(
            name="Feature2", elements=st.floats(min_value=-1000.0, max_value=1000.0)
        ),
    ],
    index=range_indexes(min_size=100, max_size=100),
)


class TestDataset:
    # @classmethod
    # def setup_class(cls):
    #     pass

    # @classmethod
    # def teardown_class(cls):
    #     pass

    # @given(df=simple_df)
    # @settings(max_examples=1)
    # def setup_method(self):
    #     self.dataset = Dataset(
    #         dataframe=df,
    #         features=["Feature1"],
    #         target="Label",
    #         task_name="Testing",
    #     )

    # def teardown_method(self):
    #     del self.dataset

    @given(df=simple_df)
    @settings(max_examples=5)
    def test_init(self, df):
        dataset = Dataset(
            dataframe=df,
            features=["Feature1"],
            target="Label",
            task_name="Testing",
        )
        assert dataset.X.columns == ["Feature1"]
        assert dataset.y.columns == ["Label"]

    @given(df=simple_df)
    @settings(max_examples=5)
    def test_split_dataset(self, df):
        dataset = Dataset(
            dataframe=df,
            features=["Feature1"],
            target="Label",
            task_name="Testing",
        )
        dataset.split_dataset()
        assert dataset.X_train.shape[0] == 60
        assert dataset.X_val.shape[0] == 20
        assert dataset.X_test.shape[0] == 20
        assert dataset.y_train.shape[0] == 60
        assert dataset.y_val.shape[0] == 20
        assert dataset.y_test.shape[0] == 20

        assert (
            dataset.X_train.shape[1]
            == dataset.X_test.shape[1]
            == dataset.X_val.shape[1]
            == 1
        )
        assert (
            dataset.y_train.shape[1]
            == dataset.y_test.shape[1]
            == dataset.y_val.shape[1]
            == 1
        )

    @given(df=simple_df)
    @settings(max_examples=5)
    def test_cross_validation_split(self, df):
        dataset = Dataset(
            dataframe=df,
            features=["Feature1"],
            target="Label",
            task_name="Testing",
        )
        dataset.cross_validation_split()
        assert dataset.X_train.shape[0] == 80
        assert dataset.X_test.shape[0] == 20
        assert dataset.y_train.shape[0] == 80
        assert dataset.y_test.shape[0] == 20
