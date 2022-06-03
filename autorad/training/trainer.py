import logging
import pickle
from pathlib import Path
from typing import Sequence

import mlflow
import numpy as np
import optuna
from optuna.integration.mlflow import MLflowCallback
from optuna.study import Study
from optuna.trial import Trial
from sklearn.metrics import roc_auc_score

from autorad.config import config
from autorad.config.type_definitions import PathLike
from autorad.data.dataset import FeatureDataset, TrainingData
from autorad.models.classifier import MLClassifier
from autorad.preprocessing.preprocessor import Preprocessor
from autorad.training import utils
from autorad.training.optimizer import OptunaOptimizer
from autorad.utils import io

log = logging.getLogger(__name__)


class Trainer:
    """
    Runs the experiment that optimizes the hyperparameters
    for all the models, given the dataset with extracted features.
    """

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
        self._optimizer = None
        self.auto_preprocessing = False
        self._init_mlflow()

    def set_optimizer(self, optimizer: str, param_fn=None, n_trials=100):
        if optimizer == "optuna":
            self._optimizer = OptunaOptimizer(
                param_fn=param_fn, n_trials=n_trials
            )
        # elif optimizer == "gridsearch":
        #     self.optimizer = GridSearchOptimizer()
        else:
            raise ValueError("Optimizer not recognized.")

    def run_auto_preprocessing(self, oversampling=True):
        selection_methods = config.FEATURE_SELECTION_METHODS
        if oversampling:
            oversampling_methods = config.OVERSAMPLING_METHODS
        else:
            oversampling_methods = [None]
        preprocessed = {}
        for selection_method in selection_methods:
            preprocessed[selection_method] = {}
            for oversampling_method in oversampling_methods:
                preprocessor = Preprocessor(
                    normalize=True,
                    feature_selection_method=selection_method,
                    oversampling_method=oversampling_method,
                )
                try:
                    preprocessed[selection_method][
                        oversampling_method
                    ] = preprocessor.fit_transform(self.dataset.data)
                except AssertionError:
                    log.error(
                        f"Preprocessing with {selection_method} and {oversampling_method} failed."
                    )
            if not preprocessed[selection_method]:
                del preprocessed[selection_method]
        with open(self.result_dir / "preprocessed.pkl", "wb") as f:
            pickle.dump(preprocessed, f, pickle.HIGHEST_PROTOCOL)

    @property
    def optimizer(self):
        if self._optimizer is None:
            raise ValueError("Optimizer is not set!")
        return self._optimizer

    def set_optuna_default_params(self, model: MLClassifier, trial: Trial):
        if self.optimizer is None:
            raise AttributeError("Optimizer not set!")
        params = self.optimizer.param_fn(model.name, trial)
        model.set_params(**params)
        return model

    def _init_mlflow(self):
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(
            "file://" + str(Path(self.registry_dir).absolute())
        )

    def run(self, auto_preprocess=False):
        """
        Run hyperparameter optimization for all the models.
        """
        mlfc = MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(), metric_name="AUC"
        )
        study = self.optimizer.create_study(study_name=self.experiment_name)
        study.optimize(
            lambda trial: self._objective(trial, auto_preprocess),
            n_trials=self.optimizer.n_trials,
            callbacks=[mlfc],
        )
        self.save_best_params(study)

    def save_best_params(self, study: Study):
        params = study.best_trial.params
        io.save_json(params, (self.result_dir / "best_params.json"))

    def _objective(self, trial: optuna.Trial, auto_preprocess=False) -> float:
        """Get params from optuna trial, return the metric."""
        if auto_preprocess:
            pkl_path = self.result_dir / "preprocessed.pkl"
            with open(pkl_path, "rb") as f:
                preprocessed = pickle.load(f)
            feature_selection_method = trial.suggest_categorical(
                "feature_selection_method", preprocessed.keys()
            )
            oversampling_method = trial.suggest_categorical(
                "oversampling_method",
                preprocessed[feature_selection_method].keys(),
            )
            data = preprocessed[feature_selection_method][oversampling_method]
        else:
            data = self.dataset.data

        model_name = trial.suggest_categorical(
            "model", [m.name for m in self.models]
        )
        model = utils.get_model_by_name(model_name, self.models)
        self.set_optuna_default_params(model, trial)
        aucs = []
        for X_train, y_train, X_val, y_val in data.iter_training():
            model.fit(X_train, y_train)
            y_pred = model.predict_proba_binary(X_val)
            auc_val = roc_auc_score(y_val, y_pred)
            aucs.append(auc_val)
        AUC = np.mean(aucs)

        return AUC


class Inferrer:
    def __init__(self, params, result_dir):
        self.params = params
        self.result_dir = result_dir
        self.model, self.preprocessor = self._parse_params()

    def _parse_params(self):
        temp_params = self.params.copy()
        selection = temp_params.pop("feature_selection_method")
        oversampling = temp_params.pop("oversampling_method")
        preprocessor = Preprocessor(
            normalize=True,
            feature_selection_method=selection,
            oversampling_method=oversampling,
        )
        model = MLClassifier.from_sklearn(temp_params.pop("model"))
        model_params = {
            "_".join(k.split("_")[1:]): v for k, v in temp_params.items()
        }
        model.set_params(**model_params)

        return model, preprocessor

    def fit(self, dataset: FeatureDataset):
        _data = self.preprocessor.fit_transform(dataset.data)
        self.model.fit(
            _data._X_preprocessed.train, _data._y_preprocessed.train
        )

    def eval(self, dataset: FeatureDataset, result_name: str = "results"):
        X = self.preprocessor.transform(dataset.data.X.test)
        y = dataset.data.y.test
        y_pred = self.model.predict_proba_binary(X)
        auc = roc_auc_score(y, y_pred)
        result = {}
        result["selected_features"] = self.preprocessor.selected_features
        result["AUC test"] = auc
        io.save_json(result, (self.result_dir / f"{result_name}.json"))
        io.save_predictions_to_csv(
            y, y_pred, (self.result_dir / f"{result_name}.csv")
        )

    def fit_eval(self, dataset: FeatureDataset, result_name: str = "results"):
        _data = self.preprocessor.fit_transform(dataset.data)
        result = {}
        result["selected_features"] = _data.selected_features
        train_auc = self._fit_eval_splits(_data)
        result["AUC train"] = train_auc
        test_auc = self._fit_eval_train_test(_data, result_name)
        result["AUC test"] = test_auc
        log.info(
            f"Test AUC: {test_auc:.3f}, mean train AUC: {np.mean(train_auc):.3f}"
        )
        io.save_json(result, (self.result_dir / f"{result_name}.json"))

    def _fit_eval_train_test(
        self, _data: TrainingData, result_name: str = "results"
    ):
        self.model.fit(
            _data._X_preprocessed.train, _data._y_preprocessed.train
        )
        y_pred = self.model.predict_proba_binary(_data._X_preprocessed.test)
        y_test = _data._y_preprocessed.test
        auc = roc_auc_score(y_test, y_pred)
        io.save_predictions_to_csv(
            y_test, y_pred, (self.result_dir / f"{result_name}.csv")
        )
        return auc

    def _fit_eval_splits(self, data: TrainingData):
        aucs = []
        for X_train, y_train, X_val, y_val in data.iter_training():
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict_proba_binary(X_val)
            auc_val = roc_auc_score(y_val, y_pred)
            aucs.append(auc_val)

        return aucs

    def init_result_df(self, dataset: FeatureDataset):
        self.result_df = dataset.meta_df.copy()
        self.test_indices = dataset.X.test.index.values
        # self.result_df = self.add_splits_to_result_df(
        #     self.result_df, self.test_indices
        # )

    # def add_splits_to_result_df(self, result_df, test_indices):
    #     result_df["test"] = 0
    #     result_df.loc[test_indices, "test"] = 1
    #     result_df["cv_split"] = -1
    #     for i in range(self.dataset.n_splits):
    #         result_df.loc[self.dataset.X_val_fold[i].index, "cv_split"] = i

    #     return result_df

    # def fit_eval_split(
    #     self, X_train, y_train, X_val, pred_colname, pred_proba_colname
    # ):
    #     # Fit and predict
    #     self.model.fit(X_train, y_train)
    #     y_pred_fold, y_pred_proba_fold = self.model.predict_label_and_proba(
    #         X_val
    #     )
    #     # Write results
    #     fold_indices = X_val.index
    #     self.result_df.loc[fold_indices, pred_colname] = y_pred_fold

    #     self.result_df.loc[
    #         fold_indices, pred_proba_colname
    #     ] = y_pred_proba_fold

    # def fit_eval_all(self):
    #     model_name = self.model.name
    #     pred_colname = f"{model_name}_pred"
    #     pred_proba_colname = f"{model_name}_pred_proba"
    #     self.result_df[pred_colname] = -1
    #     self.result_df[pred_proba_colname] = -1
    #     for i in range(self.dataset.n_splits):
    #         log.info(f"Evaluating fold: {i}")
    #         X_train_fold = self.dataset.X_train_fold[i]
    #         y_train_fold = self.dataset.y_train_fold[i]
    #         X_val_fold = self.dataset.X_val_fold[i]
    #         self.fit_eval_split(
    #             X_train_fold,
    #             y_train_fold,
    #             X_val_fold,
    #             pred_colname,
    #             pred_proba_colname,
    #         )
    #     self.fit_eval_split(
    #         self.dataset.X_train,
    #         self.dataset.y_train,
    #         self.dataset.X_test,
    #         pred_colname,
    #         pred_proba_colname,
    #     )

    # def save_results(self):
    #     df_name = f"predictions_{self.dataset.task_name}.csv"
    #     self.result_df.to_csv(self.result_dir / df_name, index=False)
    #     return self
