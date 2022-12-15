import logging

import pandas as pd
from sklearn.metrics import roc_curve

from autorad.data import FeatureDataset
from autorad.evaluation import eval_utils
from autorad.models import MLClassifier
from autorad.preprocessing import Preprocessor
from autorad.visualization import plotly_utils

log = logging.getLogger(__name__)


def evaluate_feature_dataset(
    dataset: FeatureDataset,
    model: MLClassifier,
    preprocessor: Preprocessor,
    split: str = "test",
) -> pd.DataFrame:
    """
    Evaluate a feature dataset using a model and a preprocessor.
    """
    X_preprocessed = preprocessor.transform_df(getattr(dataset.data.X, split))
    y_pred_proba = model.predict_proba_binary(X_preprocessed)
    y_true = getattr(dataset.data.y, split)

    result = pd.DataFrame(
        {
            "ID": y_true.index,
            "y_true": y_true,
            "y_pred_proba": y_pred_proba,
        }
    )

    return result


class Evaluator:
    def __init__(self, y_true, y_pred_proba):
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.threshold = eval_utils.get_youden_threshold(
            self.y_true, self.y_pred_proba
        )
        self.y_pred = self.y_pred_proba > self.threshold

    def plot_roc_curve(self):
        fig = plotly_utils.plot_roc_curve(self.y_true, self.y_pred_proba)

        return fig

    def plot_precision_recall_curve(self):
        fig = plotly_utils.plot_precision_recall_curve(
            self.y_true, self.y_pred_proba
        )

        return fig

    def plot_waterfall(self):
        fig = plotly_utils.plot_waterfall(
            self.y_true, self.y_pred_proba, self.threshold
        )
        return fig

    def plot_optimal_point_test(self, ax):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred)
        fpr_point = fpr[1]
        tpr_point = tpr[1]
        sens, spec = eval_utils.get_sensitivity_specificity(
            self.y_true, self.y_pred, self.threshold
        )
        point_label = f"Sensitivity = {sens}, Specificity = {spec}"
        ax.plot(
            [fpr_point],
            [tpr_point],
            marker="h",
            markersize=10,
            color="black",
            label=point_label,
        )
        return self
