import logging
from pathlib import Path
from typing import Sequence

import joblib
import mlflow
import numpy as np
from optuna.trial import FrozenTrial, Trial
from sklearn.metrics import roc_auc_score

from autorad.config.type_definitions import PathLike
from autorad.data import FeatureDataset, TrainingData
from autorad.models import MLClassifier
from autorad.preprocessing import Preprocessor
from autorad.training import OptunaOptimizer, train_utils
from autorad.utils import io, mlflow_utils

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
        result_dir: PathLike,
    ):
        self.dataset = dataset
        self.models = models
        self.result_dir = Path(result_dir)

        self._optimizer = None
        self.auto_preprocessing = False

    def set_optimizer(self, optimizer: str, n_trials: int = 100):
        if optimizer == "optuna":
            self._optimizer = OptunaOptimizer(n_trials=n_trials)
        else:
            raise ValueError("Optimizer not recognized.")

    @property
    def optimizer(self):
        if self._optimizer is None:
            raise ValueError("Optimizer is not set!")
        return self._optimizer

    def set_optuna_params(self, model: MLClassifier, trial: Trial):
        params = model.param_fn(trial)
        model.set_params(**params)
        return model

    def save_best_preprocessor(self, trial: FrozenTrial):
        params = trial.params
        feature_selection = params["feature_selection_method"]
        oversampling = params["oversampling_method"]
        preprocessor = Preprocessor(
            standardize=True,
            feature_selection_method=feature_selection,
            oversampling_method=oversampling,
        )
        preprocessor.fit_transform_data(self.dataset.data)
        mlflow.sklearn.log_model(preprocessor, "preprocessor")
        if "select" in preprocessor.pipeline.named_steps:
            selected_features = preprocessor.pipeline[
                "select"
            ].selected_features
            mlflow_utils.log_dict_as_artifact(
                selected_features, "selected_features"
            )

    def run(
        self,
        auto_preprocess: bool = False,
    ):
        """
        Run hyperparameter optimization for all the models.
        """
        mlflow.set_experiment("model_training")
        with mlflow.start_run():
            study = self.optimizer.create_study(
                study_name="model_training",
            )

            study.optimize(
                lambda trial: self._objective(trial, auto_preprocess),
                n_trials=self.optimizer.n_trials,
            )
            best_trial = study.best_trial
            self.log_to_mlflow(
                best_trial=best_trial,
                auto_preprocess=auto_preprocess,
            )

    def log_to_mlflow(self, best_trial: FrozenTrial, auto_preprocess: bool):
        best_model = best_trial.user_attrs["model"]
        best_auc = best_trial.user_attrs["AUC"]
        mlflow.log_metric("AUC", best_auc)
        self.save_params(best_trial)
        self.copy_extraction_artifacts()
        train_utils.log_splits(self.dataset.splits)
        self.save_best_preprocessor(best_trial)
        best_model.save_to_mlflow()

        data = self.get_trial_data(best_trial, auto_preprocess)
        train_utils.log_shap(best_model, data.X_preprocessed.train)
        self.log_train_auc(best_model, data)

    def log_train_auc(self, model: MLClassifier, data: TrainingData):
        y_true = data.y.train
        y_pred_proba = model.predict_proba_binary(data.X_preprocessed.train)
        train_auc = roc_auc_score(y_true, y_pred_proba > 0.5)
        mlflow.log_metric("train_AUC", float(train_auc))

    def copy_extraction_artifacts(self):
        try:
            extraction_run_id = self.dataset.df["extraction_ID"].iloc[0]
            mlflow_utils.copy_artifacts_from(extraction_run_id)
        except KeyError:
            log.error(
                "No extraction_id column found in feature table, copying of "
                "feature extraction params failed! This will cause problems "
                "with inference"
            )

    def save_params(self, trial):
        params = trial.params
        mlflow.log_params(params)
        io.save_json(params, (self.result_dir / "best_params.json"))

    def get_best_preprocessed_dataset(self, trial: Trial) -> TrainingData:
        """ "
        Get preprocessed dataset with preprocessing method that performed
        best in the training.
        """
        pkl_path = self.result_dir / "preprocessed.pkl"
        with open(pkl_path, "rb") as f:
            preprocessed = joblib.load(f)
        feature_selection_method = trial.suggest_categorical(
            "feature_selection_method", preprocessed.keys()
        )
        oversampling_method = trial.suggest_categorical(
            "oversampling_method",
            preprocessed[feature_selection_method].keys(),
        )
        result = preprocessed[feature_selection_method][oversampling_method]

        return result

    def get_trial_data(
        self, trial: Trial, auto_preprocess: bool = False
    ) -> TrainingData:
        """
        Get the data for the trial, either from the preprocessed data
        or from the original dataset.
        """
        if auto_preprocess:
            data = self.get_best_preprocessed_dataset(trial)
        else:
            data = self.dataset.data
        return data

    def _objective(self, trial: Trial, auto_preprocess=False) -> float:
        """Get params from optuna trial, return the metric."""
        data = self.get_trial_data(trial, auto_preprocess=auto_preprocess)

        model_name = trial.suggest_categorical(
            "model", [m.name for m in self.models]
        )
        model = train_utils.get_model_by_name(model_name, self.models)
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
            try:
                model.fit(X_train, y_train)
            except ValueError:
                log.error(f"Training {model.name} failed.")
                return np.nan
            y_pred = model.predict_proba_binary(X_val)
            auc_val = roc_auc_score(y_val, y_pred)
            aucs.append(auc_val)
        auc = float(np.mean(aucs))
        trial.set_user_attr("model", model)
        trial.set_user_attr("AUC", auc)

        return auc
