from datetime import timedelta

import hypothesis_utils
import pytest
from hypothesis import given, settings

from classrad.feature_selection.feature_selector import FeatureSelector


class TestFeatureSelector:
    @given(df=hypothesis_utils.medium_df())
    @settings(max_examples=5)
    def test_anova_selection(self, df):
        X, y = df.drop("Label", axis=1), df["Label"]
        feature_selector = FeatureSelector()
        selected_features = feature_selector.anova_selection(X, y, k=5)
        assert isinstance(selected_features, list)
        assert len(selected_features) == 5
        assert type(selected_features[0]) == str
        assert set(selected_features).issubset(set(X.columns))

    @given(df=hypothesis_utils.medium_df())
    @settings(max_examples=2, deadline=timedelta(seconds=20))
    def test_lasso_selection(self, df):
        X, y = df.drop("Label", axis=1), df["Label"]
        feature_selector = FeatureSelector()
        selected_features = feature_selector.lasso_selection(X, y)
        assert isinstance(selected_features, list)
        assert len(selected_features) >= 0
        assert set(selected_features).issubset(set(X.columns))

    @given(df=hypothesis_utils.medium_df())
    @settings(max_examples=2, deadline=timedelta(seconds=20))
    def test_boruta_selection(self, df):
        X, y = df.drop("Label", axis=1), df["Label"]
        feature_selector = FeatureSelector()
        selected_features = feature_selector.boruta_selection(X, y)
        assert isinstance(selected_features, list)
        assert len(selected_features) >= 0
        assert set(selected_features).issubset(set(X.columns))

    @given(df=hypothesis_utils.medium_df())
    @settings(max_examples=1, deadline=timedelta(seconds=20))
    def test_fit(self, df):
        X, y = df.drop("Label", axis=1), df["Label"]
        feature_selector = FeatureSelector()
        for valid_method in ["anova", "lasso", "boruta"]:
            selected_features = feature_selector.fit(X, y, method=valid_method)
            assert isinstance(selected_features, list)
            with pytest.raises(ValueError):
                feature_selector.fit(X=None, y=None, method="anova")
        with pytest.raises(ValueError):
            feature_selector = FeatureSelector()
            feature_selector.fit(X, y, method="foo")
