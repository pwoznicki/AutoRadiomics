import hypothesis_utils
import numpy as np
from hypothesis import assume, given, settings

from autorad.feature_selection.selector import (
    AnovaSelector,
    BorutaSHAPSelector,
)


class TestAnovaSelection:
    def setup_method(self):
        self.selector = AnovaSelector(n_features=5)

    @given(df=hypothesis_utils.medium_df())
    @settings(max_examples=5)
    def test_fit(self, df):
        assume(df["Label"].nunique() == 2)  # assume both categories present
        X = df.drop(columns=["Label"]).to_numpy()
        y = df["Label"].to_numpy()
        self.selector.fit(X, y)
        selected_columns = self.selector.selected_columns
        assert len(selected_columns) == 5
        assert all(item in range(X.shape[1]) for item in selected_columns)

    @given(df=hypothesis_utils.medium_df())
    def test_fit_transform(self, df):
        X = df.drop(columns=["Label"]).to_numpy()
        y = df["Label"].to_numpy()
        X_new, y = self.selector.fit_transform(X, y)
        assert isinstance(X_new, np.ndarray)
        assert X_new.shape == (X.shape[0], 5)


def test_BorutaShapSelector_fit():
    model = BorutaSHAPSelector()
    X = np.array(
        [
            np.arange(10),
            np.square(np.arange(10)),
            np.zeros(10),
        ]
    ).T
    y = np.arange(10)
    model.fit(X, y)
    assert model.selected_columns == [0, 1]
