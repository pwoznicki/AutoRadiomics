import os
from pathlib import Path

import mlflow

from autorad.config.type_definitions import PathLike


def get_model_by_name(name, models):
    for model in models:
        if model.name == name:
            return model
    raise ValueError(f"Model with name {name} not found")


def init_mlflow(registry_dir):
    registry_dir = Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("file://" + str(registry_dir.absolute()))


def mlflow_dashboard(experiment_dir: PathLike):
    command = f"mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri file://{str(experiment_dir)} &"
    os.system(command)


def log_mlflow_params(params):
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)
