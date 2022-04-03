import json
import os
from pathlib import Path

import mlflow
from optuna.study import Study

from classrad.config.type_definitions import PathLike


def get_model_by_name(name, models):
    for model in models:
        if model.name == name:
            return model
    raise ValueError(f"Model with name {name} not found")


def save_best_params(study: Study):
    params = study.best_trial.params
    print(json.dumps(params, indent=2))


def init_mlflow(experiment_name: str, registry_dir: PathLike):
    mlflow.set_tracking_uri("file://" + str(Path(registry_dir).absolute()))
    mlflow.set_experiment(experiment_name=experiment_name)


def mlflow_dashboard(experiment_dir: PathLike):
    command = f"mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri file://{str(experiment_dir)} &"
    os.system(command)


def log_mlflow_params(params):
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)
