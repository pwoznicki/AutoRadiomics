import hypothesis_utils
import pandas as pd
import pytest
from hypothesis import given, settings

from autorad.feature_selection import create_feature_selector


@pytest.mark.parametrize(
    "selection_method, kwargs",
    [
        # ("anova", {"n_features": 5}),
        ("lasso", {}),
        # ("boruta", {}),
    ],
)
@given(df=hypothesis_utils.medium_df())
@settings(max_examples=1)
def test_fit(df, selection_method, kwargs):
    selector = create_feature_selector(method=selection_method, **kwargs)
    X = df.drop(columns=["Label"])
    y = df["Label"]
    if y.nunique() == 1:
        with pytest.raises(ValueError):
            selector.fit(X, y)
    else:
        selector.fit(X, y)
        selected_features = selector.selected_features
        # assert len(selected_features) == 5
        assert all(item in X.columns for item in selected_features)
