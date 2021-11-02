"""
Create an Evaluator class to evaluate predictions for classification task
in terms of ROC AUC and sensitivity/specificity.
The models to evaluate are created on top of sklearn classifiers.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc
from Radiomics.utils.visualization import get_subplots_dimensions
from Radiomics.utils.statistics import wilcoxon_unpaired
from lofo import LOFOImportance, Dataset, plot_importance
from .metrics import roc_auc_score
from .utils import (
    get_fpr_tpr_auc,
    get_youden_threshold,
    get_sensitivity_specificity,
    common_roc_settings,
)


class Evaluator:
    def __init__(
        self,
        result_df,
        target,
        models,
        dataset,
        result_dir,
        n_jobs=1,
        random_state=None,
    ):
        self.result_df = result_df
        self.train_results = self.result_df[self.result_df["test"] == 0]
        self.test_results = self.result_df[self.result_df["test"] == 1]
        self.fold_results = self.train_results.groupby("cv_split").agg(pd.Series.tolist)
        self.target = target
        self.test_labels = self.test_results[self.target].tolist()
        self.train_labels = self.train_results[self.target].tolist()
        self.models = models
        self.dataset = dataset
        self.result_dir = result_dir
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.scores = None
        self.predictions = None
        self.predictions_proba = None
        self.best_model = None
        self.best_model_idx = None
        self.best_model_score_test = None
        self.best_model_threshold = None
        self.model_names = [model.classifier_name for model in models]

        self.result_dir.mkdir(parents=True, exist_ok=True)

    def update_predictions(self):
        self.predictions, self.predictions_proba = {}, {}
        for model in self.models:
            model_name = model.classifier_name
            self.predictions[model_name] = self.train_results[f"{model_name}_pred"]
            self.predictions_proba[model_name] = self.train_results[
                f"{model_name}_pred_proba"
            ]
        return self

    def evaluate(self):
        """
        Evaluate the models.
        """
        self.scores = {"cv": {}, "test": {}}
        for model in self.models:
            model_name = model.classifier_name
            aucs = []
            for fold in range(len(self.fold_results)):
                fold_labels = self.fold_results[self.target][fold]
                fold_preds_proba = self.fold_results[f"{model_name}_pred_proba"][fold]
                aucs.append(roc_auc_score(fold_labels, fold_preds_proba))

            model_mean_score = np.round(np.mean(aucs), 3)
            model_std_score = np.round(np.std(aucs), 3)
            self.scores["cv"][model_name] = (model_mean_score, model_std_score)
            print(
                f"For {model_name} in 5-fold CV AUC = {model_mean_score} +/- {model_std_score}"
            )
        self.update_predictions()
        self.update_best_model()
        print(
            f"Best model: {self.best_model.classifier_name} - AUC on test set = {self.best_model_score_test}"
        )

        return self

    def get_roc_threshold(self):
        y_true = self.train_labels
        y_pred_proba = self.predictions_proba[self.best_model.classifier_name]
        _, _, self.best_model_threshold = get_youden_threshold(y_true, y_pred_proba)
        return self

    def update_best_model(self):
        self.best_model_idx = np.argmax([t[0] for t in self.scores["cv"].values()])
        self.best_model = self.models[self.best_model_idx]
        model_name = self.best_model.classifier_name
        self.predictions_proba_test = self.test_results[f"{model_name}_pred_proba"]
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
        if title:
            ax.set_title(title)
        else:
            ax.set_title(model_name)
        common_roc_settings(ax)

        return self

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
        nrows, ncols, figsize = get_subplots_dimensions(len(self.models))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i, model_name in enumerate(self.model_names):
            ax = fig.axes[i]
            self.plot_roc_curve_cross_validation(model_name, ax=ax)
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(
                f"ROC Curve for {self.dataset.task_name} in 5-fold cross-validation."
            )
        fig.tight_layout()
        plt.show()
        fig.savefig(self.result_dir / "ROC.png")

        return self

    def plot_precision_recall_curve_test(self, model_name, ax=None, title=None):
        """
        Plot the precision recall curve.
        """
        y_true = self.test_labels
        y_pred_proba = self.test_results[f"{model_name}_pred_proba"]
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        auc_score = np.round(auc(recall, precision), 3)
        ax.plot(
            recall, precision, lw=2, alpha=0.8, label=f"{model_name} AUC={auc_score}"
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
        n_models = len(self.models)
        nrows, ncols, figsize = get_subplots_dimensions(n_models)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i in range(n_models):
            model_name = self.model_names[i]
            ax = fig.axes[i]
            self.plot_precision_recall_curve(model_name, ax=ax)
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(
                f"Precision-Recall Curve for {self.dataset.task_name} in test dataset"
            )
        fig.tight_layout()
        plt.show()

        return self

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
        nrows, ncols, figsize = get_subplots_dimensions(len(self.models))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i, model_name in enumerate(self.model_names):
            ax = fig.axes[i]
            self.plot_confusion_matrix_cross_validation(model_name, ax=ax)
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(
                f"Confusion Matrix for {self.dataset.task_name} in cross-validation"
            )
        plt.tight_layout()
        plt.show()
        fig.savefig(self.result_dir / "confusion_matrix.png")

        return self

    def plot_feature_importance(self, model, ax=None):
        """
        Plot importance of features for a single model
        Args:
            model [MLClassifier] - classifier
            ax (optional) - pyplot axes object
        """
        model_name = model.classifier_name
        try:
            importances = model.feature_importance()
            importance_df = pd.DataFrame(
                {"feature": self.dataset.X_train.columns, "importance": importances}
            )
            sns.barplot(x="feature", y="importance", data=importance_df, ax=ax)
            ax.set_ylabel("Feature importance")
            ax.set_title(model_name)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        except Exception:
            print(f"For {model_name} feature importance cannot be calculated.")

        return self

    def plot_feature_importance_all(self, title=None):
        """
        Plot the feature importance for all models.
        """
        nrows, ncols, figsize = get_subplots_dimensions(len(self.models))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i, model in enumerate(self.models):
            ax = fig.axes[i]
            self.plot_feature_importance(model, ax=ax)
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(f"Feature Importance for {self.dataset.task_name}")
        fig.tight_layout()
        plt.show()
        fig.savefig(self.result_dir / "feature_importance.png")

        return self

    def plot_lofo_importance(self, model):
        dataset = Dataset(
            df=self.dataset.df,
            target=self.target,
            features=self.dataset.best_features,
        )
        lofo_imp = LOFOImportance(
            dataset, model=model.classifier, scoring="neg_mean_squared_error"
        )
        importance_df = lofo_imp.get_importance()
        plot_importance(importance_df, figsize=(12, 12))
        plt.tight_layout()
        plt.show()

    def boxplot_by_class(self):
        features = self.dataset.best_features
        nrows, ncols, figsize = get_subplots_dimensions(len(features))
        fig = make_subplots(rows=nrows, cols=ncols)
        xlabels = [
            "Positive" if label == 1 else "Negative" for label in self.dataset.y_test
        ]
        xlabels = np.array(xlabels)
        # X_test = self.dataset.inverse_standardize(self.dataset.X_test)
        for i, feature in enumerate(features):
            y = self.dataset.X_test[feature]
            _, p_val = wilcoxon_unpaired(
                y[xlabels == "Negative"], y[xlabels == "Positive"]
            )
            fig.add_trace(
                go.Box(y=y, x=xlabels, name=f"{feature} p={p_val}"),
                row=i // ncols + 1,
                col=i % ncols + 1,
            )
        fig.update_layout(title_text=f"Selected features for {self.dataset.task_name}")
        fig.show()
        fig.write_html(self.result_dir / "boxplot.html")

    def plot_test(self, title=None):
        model_name = self.best_model.classifier_name

        nrows, ncols, figsize = get_subplots_dimensions(3)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        self.plot_roc_curve_test(model_name, ax=axs[0])
        self.plot_precision_recall_curve_test(model_name, ax=axs[1])
        self.plot_confusion_matrix_test(model_name, ax=axs[2])
        if title:
            fig.suptitle(title)
        else:
            fig.suptitle(
                f"Results on test dataset for {self.dataset.task_name}"
                f" using {model_name}"
            )
        fig.tight_layout()
        plt.show()
        fig.savefig(self.result_dir / "test.png")

    def plot_all_cross_validation(self):
        """
        Plot all the graphs on the cross-validation dataset.
        """
        self.plot_roc_curve_all()
        # self.plot_precision_recall_curve_all()
        self.plot_confusion_matrix_all()
        self.plot_feature_importance_all()

        return self
