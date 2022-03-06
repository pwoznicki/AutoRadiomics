import pandas as pd
import pytest

from classrad.evaluation.plotly_utils import waterfall_binary_classification


@pytest.mark.skip(reason="Plotting")
def test_waterfall_binanry_classification():
    preds = [0.1 * n for n in range(-5, 5)]
    labels = [str(n % 2) for n in range(10)]
    df = pd.DataFrame({"Prediction": preds, "Label": labels})
    waterfall_binary_classification(
        df, prediction_colname="Prediction", label_colname="Label"
    )
