import hypothesis_utils
import pytest
from hypothesis import assume, given, settings

from autorad.visualization.plotly_utils import (
    boxplot_by_class,
    plot_roc_curve,
    waterfall_binary_classification,
)


@pytest.mark.skip(reason="Plotting")
def test_waterfall_binary_classification():
    pred_probas = [0.1 * n for n in range(10)]
    labels = [(n % 2) for n in range(10)]
    fig = waterfall_binary_classification(labels, pred_probas, 0.5)
    fig.show()


@pytest.mark.skip(reason="Plotting")
def test_plot_roc_curve():
    pred_probas = [0.1 * n for n in range(10)]
    labels = [(n % 2) for n in range(10)]
    fig = plot_roc_curve(labels, pred_probas)
    fig.show()


@pytest.mark.skip(reason="Plotting")
@given(df=hypothesis_utils.medium_df())
@settings(max_examples=3, deadline=None)
def test_boxplot_by_class(df):
    X = df.drop(columns=["Label"])
    y = df["Label"]
    assume(y.nunique() == 2)
    fig = boxplot_by_class(X, y)
    fig.show()
