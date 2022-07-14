import logging

from sklearn.metrics import roc_curve

from autorad.visualization.plotly_utils import plot_roc_curve, plot_waterfall

from .utils import get_sensitivity_specificity, get_youden_threshold

log = logging.getLogger(__name__)


class SimpleEvaluator:
    def __init__(self, y_true, y_pred_proba):
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.threshold = get_youden_threshold(self.y_true, self.y_pred_proba)
        self.y_pred = self.y_pred_proba > self.threshold

    def plot_roc_curve(self):
        fig = plot_roc_curve(self.y_true, self.y_pred_proba)

        return fig


    def plot_waterfall(self):
        fig = plot_waterfall(self.y_true, self.y_pred_proba, self.threshold)
        return fig

    def plot_optimal_point_test(self, ax):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred)
        fpr_point = fpr[1]
        tpr_point = tpr[1]
        sens, spec = get_sensitivity_specificity(
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
