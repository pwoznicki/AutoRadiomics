import os
from pathlib import Path
from typing import List

import lofo
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import roc_auc_score

from classrad.config.type_definitions import PathLike
from classrad.data.dataset import FeatureDataset
from classrad.feature_selection.feature_selector import FeatureSelector
from classrad.models.classifier import MLClassifier
from classrad.training.optimizer import GridSearchOptimizer
from classrad.visualization.visualization import get_subplots_dimensions


class Trainer:
    def __init__(
        self,
        dataset: FeatureDataset,
        models: List[MLClassifier],
        result_dir: PathLike,
        feature_selection: str = "lasso",
        num_features: int = 10,
        experiment_name: str = "baseline",
    ):
        self.dataset = dataset
        self.models = models
        self.result_dir = Path(result_dir)
        self.feature_selection = feature_selection
        self.num_features = num_features
        self.experiment_name = experiment_name
        self.model_names = [model.name for model in models]
        self.test_indices = None

        self.result_dir.mkdir(parents=True, exist_ok=True)

    def _init_mlflow(self):
        # with mlflow.start_run(nested=True) as run:  # NOQA: F841
        #     run_id = mlflow.active_run().info.run_id
        #     print(f"MLflow run id: {run_id}")
        mlflow.set_experiment(experiment_name=self.experiment_name)

    def _mlflow_dashboard(self):
        os.system(
            "mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri $PWD/experiments/ &"
        )

    def _log_mlflow_params(self, params):
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

    def _optimize_cross_validation_single_model(self, model):
        print(f"Training and inferring model: {model.name}")
        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(), metric_name="AUC"
        )
        study = model.optimizer.study
        study.optimize(
            lambda trial: self._objective(trial, model),
            n_trials=model.optimizer.n_trials,
            callbacks=[mlflow_callback],
        )

        best_hyperparams = study.best_trial.params
        print(f"Best hyperparameters: {best_hyperparams}")
        return best_hyperparams

    def optimize_cross_validation(self):
        """
        Optimize all the models.
        """
        # self.init_result_df()
        self._init_mlflow()
        # self._mlflow_dashboard()
        self._standardize_and_select_features()
        for model in self.models:
            best_hyperparams = self._optimize_cross_validation_single_model(
                model
            )
            self._log_mlflow_params(
                {
                    "model": model,
                    "feature selection method": self.feature_selection,
                    "features": self.dataset.best_features,
                    "best_hyperparams": best_hyperparams,
                }
            )

        return self

    def _objective(self, trial, model):
        params = model.optimizer.param_fn(trial)
        model.set_params(**params)
        model.fit(self.dataset.X_train_fold[0], self.dataset.y_train_fold[0])
        y_pred_val = model.predict_proba(self.dataset.X_val_fold[0])[:, 1]
        auc_val = roc_auc_score(self.dataset.y_val_fold[0], y_pred_val)

        return auc_val

    def _standardize_and_select_features(self):
        self.dataset.standardize_features()
        feature_selector = FeatureSelector()
        self.dataset = feature_selector.fit_transform_dataset(
            self.dataset,
            method=self.feature_selection,
            k=self.num_features,
        )
        return self

    def _tune_sklearn_gridsearch(self, model):
        optimizer = GridSearchOptimizer(
            dataset=self.dataset,
            model=model,
            param_dir=self.result_dir / "optimal_params",
        )
        model = optimizer.load_or_tune_hyperparameters()

    def plot_feature_importance(self, model, ax=None):
        """
        Plot importance of features for a single model
        Args:
            model [MLClassifier] - classifier
            ax (optional) - pyplot axes object
        """
        model_name = model.name
        try:
            importances = model.feature_importance()
            importance_df = pd.DataFrame(
                {
                    "feature": self.dataset.X_train.columns,
                    "importance": importances,
                }
            )
            sns.barplot(x="feature", y="importance", data=importance_df, ax=ax)
            ax.tick_params(axis="both", labelsize="x-small")
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
        fig.savefig(
            self.result_dir / "feature_importance.png",
            bbox_inches="tight",
            dpi=100,
        )
        plt.show()

        return self

    def plot_lofo_importance(self, model):
        dataset = lofo.Dataset(
            df=self.dataset.df,
            target=self.target,
            features=self.dataset.best_features,
        )
        lofo_imp = lofo.LOFOImportance(
            dataset, model=model.classifier, scoring="neg_mean_squared_error"
        )
        importance_df = lofo_imp.get_importance()
        lofo.plot_importance(importance_df, figsize=(12, 12))
        plt.tight_layout()
        plt.show()


class Inferrer:
    def __init__(self, dataset, model, result_dir):
        self.dataset = dataset
        self.model = model
        self.result_df = None
        self.result_dir = result_dir

    def init_result_df(self):
        self.result_df = self.dataset.meta_df.copy()
        self.test_indices = self.dataset.X_test.index.values
        self.result_df = self.add_splits_to_result_df(
            self.result_df, self.test_indices
        )

    def add_splits_to_result_df(self, result_df, test_indices):
        result_df["test"] = 0
        result_df.loc[test_indices, "test"] = 1
        result_df["cv_split"] = -1
        for i in range(self.dataset.n_splits):
            result_df.loc[self.dataset.X_val_fold[i].index, "cv_split"] = i

        return result_df

    def fit_eval_split(
        self, X_train, y_train, X_val, pred_colname, pred_proba_colname
    ):
        # Fit and predict
        self.model.fit(X_train, y_train)
        y_pred_fold, y_pred_proba_fold = self.model.predict_label_and_proba(
            X_val
        )
        # Write results
        fold_indices = X_val.index
        self.result_df.loc[fold_indices, pred_colname] = y_pred_fold

        self.result_df.loc[
            fold_indices, pred_proba_colname
        ] = y_pred_proba_fold

    def fit_eval_all(self):
        model_name = self.model.name
        pred_colname = f"{model_name}_pred"
        pred_proba_colname = f"{model_name}_pred_proba"
        self.result_df[pred_colname] = -1
        self.result_df[pred_proba_colname] = -1
        for i in range(self.dataset.n_splits):
            print(f"Evaluating fold: {i}")
            X_train_fold = self.dataset.X_train_fold[i]
            y_train_fold = self.dataset.y_train_fold[i]
            X_val_fold = self.dataset.X_val_fold[i]
            self.fit_eval_split(
                X_train_fold,
                y_train_fold,
                X_val_fold,
                pred_colname,
                pred_proba_colname,
            )
        self.fit_eval_split(
            self.dataset.X_train,
            self.dataset.y_train,
            self.dataset.X_test,
            pred_colname,
            pred_proba_colname,
        )

    def save_results(self):
        df_name = f"predictions_{self.dataset.task_name}.csv"
        self.result_df.to_csv(self.result_dir / df_name, index=False)
        return self
