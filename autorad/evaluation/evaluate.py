import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from autorad.data import FeatureDataset
from autorad.evaluation import eval_utils
from autorad.models import MLClassifier
from autorad.preprocessing import Preprocessor
from autorad.visualization.plotly_utils import plot_roc_curve, plot_waterfall

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
        fig = plot_roc_curve(self.y_true, self.y_pred_proba)

        return fig

    def plot_precision_recall_curve(self, title=None):
        """
        Plot the precision recall curve.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        precision, recall, _ = precision_recall_curve(
            self.y_true, self.y_pred_proba
        )
        auc_score = np.round(auc(recall, precision), 3)
        ax.plot(
            recall,
            precision,
            lw=2,
            alpha=0.8,
            label=f"AUC={auc_score}",
        )
        ax.set_xlabel("Recall", fontsize=40)
        ax.set_ylabel("Precision", fontsize=40)
        ax.legend(loc="lower left", fontsize=40)
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Precision-Recall Curve", fontsize=40)
        fig.tight_layout()

        return fig

    def plot_waterfall(self):
        fig = plot_waterfall(self.y_true, self.y_pred_proba, self.threshold)
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
