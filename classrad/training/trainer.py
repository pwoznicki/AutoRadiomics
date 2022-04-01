import logging
from pathlib import Path
from typing import Sequence

import mlflow
import numpy as np
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import roc_auc_score

from classrad.config import config
from classrad.config.type_definitions import PathLike
from classrad.data.dataset import FeatureDataset
from classrad.models.classifier import MLClassifier

log = logging.getLogger(__name__)


class ModelSubtrainer:
    def __init__(
        self, dataset: FeatureDataset, model: MLClassifier, mlflow_callback
    ):
        self.dataset = dataset
        self.model = model
        self.optimizer = model.optimizer
        self.mlflow_callback = mlflow_callback

    def run(self):
        log.info(f"Training and inferring model: {self.model.name}")
        study = self.model.optimizer.create_study(study_name=self.model.name)
        study.optimize(
            lambda trial: self._objective(trial),
            n_trials=self.optimizer.n_trials,
            callbacks=[self.mlflow_callback],
        )

        best_hyperparams = study.best_trial.params
        log.info(f"Best hyperparameters: {best_hyperparams}")
        return best_hyperparams

    def _objective(self, trial):
        X = self.dataset.data._X_preprocessed
        y = self.dataset.data._y_preprocessed

        self.model.set_optuna_default_params(trial)
        aucs = []
        for i in range(len(self.dataset.cv_splits)):
            self.model.fit(X.train_folds[i], y.train_folds[i])
            y_pred = self.model.predict_proba_binary(X.val_folds[i])
            auc_val = roc_auc_score(y.val_folds[i], y_pred)
            aucs.append(auc_val)
        AUC = np.mean(aucs)
        # utils.log_mlflow_params(
        #     {
        #         "features": self.dataset.data.selected_features,
        #     }
        # )
        # mlflow.sklearn.log_model(self.model, "model")

        return AUC


class Trainer:
    def __init__(
        self,
        dataset: FeatureDataset,
        models: Sequence[MLClassifier],
        result_dir: PathLike = config.RESULT_DIR,
        experiment_name: str = "baseline",
    ):
        self.dataset = dataset
        self.models = models
        self.result_dir = Path(result_dir)
        self.experiment_name = experiment_name

        self.registry_dir = self.result_dir / "models"
        self.experiment_dir = self.result_dir / "experiments"
        self._init_mlflow()

    def _init_mlflow(self):
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(
            "file://" + str(Path(self.registry_dir).absolute())
        )
        mlflow.set_experiment(experiment_name=self.experiment_name)
        self.mlflow_callback = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(), metric_name="AUC"
        )
        return self.mlflow_callback

    def run(self):
        """
        Run hyperparameter optimization for all the models.
        """
        for model in self.models:
            subtrainer = ModelSubtrainer(
                dataset=self.dataset,
                model=model,
                mlflow_callback=self.mlflow_callback,
            )
            subtrainer.run()

    # def _tune_sklearn_gridsearch(self, model):
    #     optimizer = GridSearchOptimizer(
    #         dataset=self.dataset,
    #         model=model,
    #         param_dir=self.result_dir / "optimal_params",
    #     )
    #     model = optimizer.load_or_tune_hyperparameters()


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
            log.info(f"Evaluating fold: {i}")
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
