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
from autorad.data.dataset import FeatureDataset
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
        self._optimizer = None
        self.auto_preprocessing = False
        utils.init_mlflow(self.registry_dir)

    def set_optimizer(self, optimizer: str, n_trials=100):
        if optimizer == "optuna":
            self._optimizer = OptunaOptimizer(n_trials=n_trials)
        # elif optimizer == "gridsearch":
        #     self.optimizer = GridSearchOptimizer()
        else:
            raise ValueError("Optimizer not recognized.")

    def run_auto_preprocessing(
        self, oversampling=True, selection_methods=None
    ):
        if selection_methods is None:
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

    def set_optuna_params(self, model: MLClassifier, trial: Trial):
        params = model.param_fn(trial)
        model.set_params(**params)
        return model

    def run(
        self,
        auto_preprocess: bool = False,
    ):
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

    def optimize_preprocessing(self, trial: Trial):
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
        result = preprocessed[feature_selection_method][oversampling_method]

        return result

    def _objective(self, trial: optuna.Trial, auto_preprocess=False) -> float:
        """Get params from optuna trial, return the metric."""
        if auto_preprocess:
            data = self.optimize_preprocessing(trial)
        else:
            data = self.dataset.data

        model_name = trial.suggest_categorical(
            "model", [m.name for m in self.models]
        )
        model = utils.get_model_by_name(model_name, self.models)
        model = self.set_optuna_params(model, trial)
        aucs = []
        for (
            X_train,
            y_train,
            _,
            X_val,
            y_val,
            _,
        ) in data.iter_training():
            model.fit(X_train, y_train)
            y_pred = model.predict_proba_binary(X_val)
            auc_val = roc_auc_score(y_val, y_pred)
            aucs.append(auc_val)
        AUC = np.mean(aucs)

        return AUC
