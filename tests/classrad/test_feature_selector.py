import pytest
from datetime import timedelta
from hypothesis.extra.pandas import data_frames, column, range_indexes
from hypothesis import given, settings, strategies as st

from classrad.feature_selection.feature_selector import FeatureSelector


def create_hypothesis_df():
    feature_columns = [
        column(
            name=f"Feature_{i}",
            elements=st.floats(min_value=0.0, max_value=1.0),
        )
        for i in range(10)
    ]
    label_column = column(name="Label", elements=st.booleans())
    all_columns = feature_columns + [label_column]
    df = data_frames(
        columns=all_columns,
        index=range_indexes(min_size=5, max_size=10),
    )
    return df


class TestFeatureSelector:
    @given(df=create_hypothesis_df())
    @settings(max_examples=5)
    def test_anova_selection(self, df):
        X, y = df.drop("Label", axis=1), df["Label"]
        feature_selector = FeatureSelector()
        feature_selector.anova_selection(X, y, k=5)
        assert isinstance(feature_selector.best_features, list)
        assert len(feature_selector.best_features) == 5
        assert type(feature_selector.best_features[0]) == str
        assert set(feature_selector.best_features).issubset(set(X.columns))

    @given(df=create_hypothesis_df())
    @settings(max_examples=2, deadline=timedelta(seconds=20))
    def test_lasso_selection(self, df):
        X, y = df.drop("Label", axis=1), df["Label"]
        feature_selector = FeatureSelector()
        feature_selector.lasso_selection(X, y)
        assert isinstance(feature_selector.best_features, list)
        assert len(feature_selector.best_features) >= 0
        assert set(feature_selector.best_features).issubset(set(X.columns))

    @given(df=create_hypothesis_df())
    @settings(max_examples=2, deadline=timedelta(seconds=20))
    def test_boruta_selection(self, df):
        X, y = df.drop("Label", axis=1), df["Label"]
        feature_selector = FeatureSelector()
        feature_selector.boruta_selection(X, y)
        assert isinstance(feature_selector.best_features, list)
        assert len(feature_selector.best_features) >= 0
        assert set(feature_selector.best_features).issubset(set(X.columns))

    @given(df=create_hypothesis_df())
    @settings(max_examples=1, deadline=timedelta(seconds=20))
    def test_fit(self, df):
        X, y = df.drop("Label", axis=1), df["Label"]
        feature_selector = FeatureSelector()
        for valid_method in ["anova", "lasso", "boruta"]:
            feature_selector.fit(X, y, method=valid_method)
            assert isinstance(feature_selector.best_features, list)
            with pytest.raises(ValueError):
                feature_selector.fit(X=None, y=None, method="anova")
        with pytest.raises(ValueError):
            feature_selector = FeatureSelector()
            feature_selector.fit(X, y, method="foo")
