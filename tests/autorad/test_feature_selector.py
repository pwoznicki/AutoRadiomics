import hypothesis_utils
import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings

from autorad.feature_selection.selector import (
    AnovaSelector,
    create_feature_selector,
)


class TestAnovaSelection:
    def setup_method(self):
        self.selector = AnovaSelector(n_features=5)

    @given(df=hypothesis_utils.medium_df())
    @settings(max_examples=5)
    def test_fit(self, df):
        assume(df["Label"].nunique() == 2)  # assume both categories present
        X = df.drop(columns=["Label"])
        y = df["Label"]
        self.selector.fit(X, y)
        selected_features = self.selector.selected_features
        assert len(selected_features) == 5
        assert all(item in X.columns for item in selected_features)

    @given(df=hypothesis_utils.medium_df())
    def test_fit_transform(self, df):
        X = df.drop(columns=["Label"])
        y = df["Label"]
        X_new, y = self.selector.fit_transform(X, y)
        assert isinstance(X_new, pd.DataFrame)
        assert X_new.shape == (X.shape[0], 5)
