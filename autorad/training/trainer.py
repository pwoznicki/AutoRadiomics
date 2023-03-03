import logging
from pathlib import Path
from typing import Sequence

import joblib
import mlflow
import numpy as np
from optuna.trial import Trial
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
        seed: int = 42,
    ):
        self.dataset = dataset
        self.models = models
        self.result_dir = Path(result_dir)
        self.seed = seed

        self._optimizer = None
        self.auto_preprocessing = False

    def set_optimizer(self, optimizer: str, n_trials: int = 100):
        if optimizer == "optuna":
            self._optimizer = OptunaOptimizer(
                n_trials=n_trials, seed=self.seed
            )
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

    def save_best_preprocessor(self, best_trial_params: dict):
        feature_selection = best_trial_params["feature_selection_method"]
        oversampling = best_trial_params["oversampling_method"]
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
        experiment_name="model_training",
    ):
        """
        Run hyperparameter optimization for all the models.
        """
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(experiment_name)
        else:
            log.warn("Running training in existing experiment.")
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            study = self.optimizer.create_study(
                study_name=experiment_name,
            )

            study.optimize(
                lambda trial: self._objective(
                    trial, auto_preprocess=auto_preprocess
                ),
                n_trials=self.optimizer.n_trials,
                callbacks=[_save_model_callback],
            )
            self.log_to_mlflow(study=study)

    def log_to_mlflow(self, study):
        best_auc = study.user_attrs["AUC_val"]
        mlflow.log_metric("AUC_val", best_auc)

        best_model = study.user_attrs["model"]
        best_model.save_to_mlflow()

        best_params = study.best_trial.params
        self.save_params(best_params)
        self.save_best_preprocessor(best_params)
        self.copy_extraction_artifacts()
        train_utils.log_dataset(self.dataset)
        train_utils.log_splits(self.dataset.splits)

        data_preprocessed = study.user_attrs["data_preprocessed"]
        train_utils.log_shap(best_model, data_preprocessed.X.train)
        self.log_train_auc(best_model, data_preprocessed)

    def log_train_auc(self, model: MLClassifier, data: TrainingData):
        y_true = data.y.train
        y_pred_proba = model.predict_proba_binary(data.X.train)
        auc_train = roc_auc_score(y_true, y_pred_proba)
        mlflow.log_metric("AUC_train", float(auc_train))

    def copy_extraction_artifacts(self):
        try:
            extraction_run_id = self.dataset.df["extraction_ID"].iloc[0]
            mlflow_utils.copy_artifacts_from(extraction_run_id)
        except KeyError:
            log.warn(
                "Copying of feature extraction params failed! "
                "No extraction_id column found in feature table. "
                "This will cause problems with inference from images."
            )
        except mlflow.exceptions.MlflowException:
            log.warn(
                "Copying of feature extraction params failed! "
                "No feature extraction artifact included in the run. "
                "This will cause problems with inference from images."
            )

    def save_params(self, params: dict):
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
        model = self.set_optuna_params(model=model, trial=trial)
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
            y_pred_proba = model.predict_proba_binary(X_val)
            auc_val = roc_auc_score(y_val, y_pred_proba)

            aucs.append(auc_val)
        model.fit(
            data.X.train, data.y.train
        )  # refit on the whole training set (important for cross-validation)
        auc_val = float(np.mean(aucs))
        trial.set_user_attr("AUC_val", auc_val)
        trial.set_user_attr("model", model)
        trial.set_user_attr("data_preprocessed", data)

        return auc_val


def _save_model_callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="AUC_val", value=trial.user_attrs["AUC_val"])
        study.set_user_attr(key="model", value=trial.user_attrs["model"])
        study.set_user_attr(
            key="data_preprocessed",
            value=trial.user_attrs["data_preprocessed"],
        )
