import pytest

from autorad.visualization.plotly_utils import (
    plot_ROC,
    waterfall_binary_classification,
)


@pytest.mark.skip(reason="Plotting")
def test_waterfall_binary_classification():
    preds = [0.1 * n for n in range(10)]
    labels = [(n % 2) for n in range(10)]
    fig = waterfall_binary_classification(labels, preds, 0.5)
    fig.show()


def test_plot_ROC():
    preds = [0.1 * n for n in range(10)]
    labels = [(n % 2) for n in range(10)]
    fig = plot_ROC(labels, preds)
    fig.show()
