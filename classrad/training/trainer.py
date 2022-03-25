from pathlib import Path
from typing import List

import mlflow
import numpy as np
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import roc_auc_score

from classrad.config import config
from classrad.config.type_definitions import PathLike
from classrad.data.dataset import FeatureDataset
from classrad.models.classifier import MLClassifier
from classrad.training.optimizer import GridSearchOptimizer

from . import utils


class Trainer:
    def __init__(
        self,
        dataset: FeatureDataset,
        models: List[MLClassifier],
        result_dir: PathLike = config.RESULT_DIR,
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

        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.registry_dir = self.result_dir / "models"
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir = self.result_dir / "experiments"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def _optimize_single_model(self, model: MLClassifier):
        print(f"Training and inferring model: {model.name}")
        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(), metric_name="AUC"
        )
        study = model.optimizer.create_study()
        study.optimize(
            lambda trial: self._objective(trial, model),
            n_trials=model.optimizer.n_trials,
            callbacks=[mlflow_callback],
        )

        best_hyperparams = study.best_trial.params
        print(f"Best hyperparameters: {best_hyperparams}")
        return best_hyperparams

    def run(self):
        """
        Run hyperparameter optimization for all the models.
        """
        utils.init_mlflow(self.experiment_name, self.registry_dir)
        utils.mlflow_dashboard(self.experiment_dir)
        self._normalize_and_select_features()
        for model in self.models:
            best_hyperparams = self._optimize_single_model(model)
            utils.log_mlflow_params(
                {
                    "model": model,
                    "feature selection method": self.feature_selection,
                    "features": self.dataset.data.selected_features,
                    "best_hyperparams": best_hyperparams,
                }
            )
            mlflow.sklearn.log_model(model, "model")

        return self

    def _objective(self, trial, model: MLClassifier):
        X = self.dataset.data.X_selected
        y = self.dataset.data.y

        assert X.train_folds is not None
        assert y.train_folds is not None
        assert X.val_folds is not None
        assert y.val_folds is not None

        params = model.optimizer.param_fn(trial)
        model.set_params(**params)
        aucs = []
        for X_train, y_train, X_val, y_val in zip(
            X.train_folds, y.train_folds, X.val_folds, y.val_folds
        ):
            model.fit(X_train, y_train)
            y_pred = model.predict_proba_binary(X_val)
            auc_val = roc_auc_score(y_val, y_pred)
            aucs.append(auc_val)
        AUC = np.mean(aucs)
        trial.set_user_attr("AUC", AUC)

        return AUC

    def _normalize_and_select_features(self):
        self.dataset.data.normalize_features()
        self.dataset.data.select_features(
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
