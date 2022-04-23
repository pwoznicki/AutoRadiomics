"""
Create an Evaluator class to evaluate predictions for classification task
in terms of ROC AUC and sensitivity/specificity.
The models to evaluate are created on top of sklearn classifiers.
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from autorad.visualization.matplotlib_utils import (
    common_roc_settings,
    get_subplots_dimensions,
)
from autorad.visualization.plotly_utils import waterfall_binary_classification

from .metrics import roc_auc_score
from .utils import (
    get_fpr_tpr_auc,
    get_sensitivity_specificity,
    get_youden_threshold,
)

log = logging.getLogger(__name__)


class SimpleEvaluator:
    def __init__(self, y_true, y_pred_proba):
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        _, _, self.threshold = get_youden_threshold(
            self.y_true, self.y_pred_proba
        )
        self.y_pred = self.y_pred_proba > self.threshold

    def plot_roc_curve(self, title=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        fpr, tpr, roc_auc = get_fpr_tpr_auc(self.y_true, self.y_pred_proba)
        label = f"AUC = {roc_auc}"
        ax.plot(fpr, tpr, lw=3, alpha=0.8, label=label)
        self.plot_optimal_point_test(ax)
        if title:
            ax.set_title(title)
        else:
            ax.set_title("ROC Curve")
        common_roc_settings(ax, fontsize=40)
        fig.tight_layout()
        plt.show()

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
        fig = waterfall_binary_classification(
            self.y_true, self.y_pred_proba, self.threshold
        )
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


class Evaluator:
    def __init__(
        self,
        result_df,
        target,
        result_dir,
        n_jobs=1,
        random_state=None,
    ):
        self.result_df = result_df
        self.train_results = self.result_df[self.result_df["test"] == 0]
        self.test_results = self.result_df[self.result_df["test"] == 1]
        self.fold_results = self.train_results.groupby("cv_split").agg(
            pd.Series.tolist
        )
        self.target = target
        self.test_labels = self.test_results[self.target].tolist()
        self.train_labels = self.train_results[self.target].tolist()
        self.result_dir = result_dir
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.scores = None
        self.predictions = None
        self.predictions_proba = None
        self.model_names = None
        self.best_model_name = None
        self.best_model_idx = None
        self.best_model_score_test = None
        self.best_model_threshold = None

        self.result_dir.mkdir(parents=True, exist_ok=True)

    def update_model_names(self):
        colnames = self.result_df.columns.tolist()
        relevant_colnames = [
            name for name in colnames if name.endswith("pred")
        ]
        self.model_names = [
            colname.split("_")[0] for colname in relevant_colnames
        ]
        return self

    def update_predictions(self):
        self.predictions, self.predictions_proba = {}, {}
        for model_name in self.model_names:
            self.predictions[model_name] = self.train_results[
                f"{model_name}_pred"
            ]
            self.predictions_proba[model_name] = self.train_results[
                f"{model_name}_pred_proba"
            ]
        return self

    def evaluate(self):
        """
        Evaluate the models.
        """
        self.update_model_names()
        self.scores = {"cv": {}, "test": {}}
        for model_name in self.model_names:
            aucs = []
            for fold in range(len(self.fold_results)):
                fold_labels = self.fold_results[self.target][fold]
                fold_preds_proba = self.fold_results[
                    f"{model_name}_pred_proba"
                ][fold]
                aucs.append(roc_auc_score(fold_labels, fold_preds_proba))

            model_mean_score = np.round(np.mean(aucs), 3)
            model_std_score = np.round(np.std(aucs), 3)
            self.scores["cv"][model_name] = (model_mean_score, model_std_score)
            log.info(
                f"For {model_name} in 5-fold CV AUC = {model_mean_score} \
                  +/- {model_std_score}"
            )
        self.update_predictions()
        self.update_best_model()
        log.info(
            f"Best model: {self.best_model_name} - AUC on test set = \
              {self.best_model_score_test}"
        )

        return self

    def get_roc_threshold(self):
        y_true = self.train_labels
        y_pred_proba = self.predictions_proba[self.best_model_name]
        _, _, self.best_model_threshold = get_youden_threshold(
            y_true, y_pred_proba
        )
        return self

    def update_best_model(self):
        self.best_model_idx = np.argmax(
            [t[0] for t in self.scores["cv"].values()]
        )
        self.best_model_name = self.model_names[self.best_model_idx]
        self.predictions_proba_test = self.test_results[
            f"{self.best_model_name}_pred_proba"
        ]
        self.best_model_score_test = roc_auc_score(
            self.test_labels, self.predictions_proba_test
        )
        self.get_roc_threshold()
        return self

    def plot_roc_curve_cross_validation(self, model_name, ax, title=None):
        """
        Plot the ROC curve.
        """
        y_true = self.train_labels
        y_pred_proba = self.predictions_proba[model_name]
        (auc_mean, auc_std) = self.scores["cv"][model_name]
        fpr, tpr, roc_auc = get_fpr_tpr_auc(y_true, y_pred_proba)
        label = f"Cumulative AUC={roc_auc}, mean AUC={auc_mean}+/-{auc_std}"
        ax.plot(fpr, tpr, lw=3, alpha=0.8, label=label)
        # ax.legend(label, fontsize=10)
        if title:
            ax.set_title(title)
        else:
            ax.set_title(model_name)
        common_roc_settings(ax)

        return ax

    def plot_optimal_point_test(self, y_true, y_pred_proba, ax):
        thr = self.best_model_threshold
        y_pred = y_pred_proba > thr
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        fpr_point = fpr[1]
        tpr_point = tpr[1]
        sens, spec = get_sensitivity_specificity(y_true, y_pred, thr)
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

    def plot_roc_curve_test(self, model_name, ax, title=None):
        """
        Plot the ROC curve.
        """
        y_true = self.test_labels
        y_pred_proba = self.test_results[f"{model_name}_pred_proba"]
        fpr, tpr, roc_auc = get_fpr_tpr_auc(y_true, y_pred_proba)
        label = f"{model_name} - AUC = {roc_auc}"
        ax.plot(fpr, tpr, lw=3, alpha=0.8, label=label)
        self.plot_optimal_point_test(y_true, y_pred_proba, ax)
        if title:
            ax.set_title(title)
        else:
            ax.set_title("ROC Curve")
        common_roc_settings(ax)

        return self

    def plot_roc_curve_all(self, title=None):
        """
        Plot the ROC Curve for all models.
        """
        nrows, ncols, figsize = get_subplots_dimensions(len(self.model_names))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i, model_name in enumerate(self.model_names):
            ax = fig.axes[i]
            fig.axes[i] = self.plot_roc_curve_cross_validation(
                model_name, ax=ax
            )
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(
                "ROC Curve for training set in 5-fold cross-validation."
            )
        fig.tight_layout()
        plt.show()
        fig.savefig(
            self.result_dir / "ROC.png", bbox_inches="tight", dpi=fig.dpi
        )

        return fig

    def plot_precision_recall_curve_test(
        self, model_name, ax=None, title=None
    ):
        """
        Plot the precision recall curve.
        """
        y_true = self.test_labels
        y_pred_proba = self.test_results[f"{model_name}_pred_proba"]
        precision, recall, thresholds = precision_recall_curve(
            y_true, y_pred_proba
        )
        auc_score = np.round(auc(recall, precision), 3)
        ax.plot(
            recall,
            precision,
            lw=2,
            alpha=0.8,
            label=f"{model_name} AUC={auc_score}",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="lower left")
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Precision-Recall Curve")

        return self

    def plot_precision_recall_curve_all(self, title=None):
        """
        Plot the precision recall curve for all models.
        """
        n_models = len(self.model_names)
        nrows, ncols, figsize = get_subplots_dimensions(n_models)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i in range(n_models):
            model_name = self.model_names[i]
            ax = fig.axes[i]
            self.plot_precision_recall_curve(model_name, ax=ax)
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle("Precision-Recall Curve for test dataset")
        fig.tight_layout()
        plt.show()

        return fig

    def plot_confusion_matrix_cross_validation(self, model_name, ax=None):
        """
        Plot the confusion matrix for a single model.
        """
        y_true = self.train_labels
        y_pred = self.predictions[model_name]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(model_name)

        return self

    def plot_confusion_matrix_test(self, model_name, ax=None, title=None):
        """
        Plot the confusion matrix for a single model.
        """
        y_true = self.test_labels
        y_pred = self.test_results[f"{model_name}_pred"]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        if title:
            ax.set_title(title)
        else:
            ax.set_title("Confusion Matrix")

        return self

    def plot_confusion_matrix_all(self, title=None):
        """
        Plot the confusion matrix for all models.
        """
        nrows, ncols, figsize = get_subplots_dimensions(len(self.model_names))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i, model_name in enumerate(self.model_names):
            ax = fig.axes[i]
            self.plot_confusion_matrix_cross_validation(model_name, ax=ax)
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle("Confusion Matrix for 5-fold cross-validation")
        plt.tight_layout()
        plt.show()
        fig.savefig(
            self.result_dir / "confusion_matrix.png", bbox_inches="tight"
        )

        return fig

    def plot_test(self, title=None):
        model_name = self.best_model_name

        nrows, ncols, figsize = get_subplots_dimensions(3)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        self.plot_roc_curve_test(model_name, ax=axs[0])
        self.plot_precision_recall_curve_test(model_name, ax=axs[1])
        self.plot_confusion_matrix_test(model_name, ax=axs[2])
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(f"Results on test dataset using {model_name}")
        fig.tight_layout()
        plt.show()
        fig.savefig(self.result_dir / "test.png", bbox_inches="tight")

        return fig

    def plot_all_cross_validation(self):
        """
        Plot all the graphs on the cross-validation dataset.
        """
        self.plot_roc_curve_all()
        # self.plot_precision_recall_curve_all()
        self.plot_confusion_matrix_all()

        return self
