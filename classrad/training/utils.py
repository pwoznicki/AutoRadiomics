import os
from pathlib import Path

import mlflow

from classrad.config import config
from classrad.config.type_definitions import PathLike


def init_mlflow(
    experiment_name: str, registry_dir: PathLike = config.MODEL_REGISTRY
):
    # with mlflow.start_run(nested=True) as run:  # NOQA: F841
    #     run_id = mlflow.active_run().info.run_id
    #     print(f"MLflow run id: {run_id}")
    mlflow.set_tracking_uri("file://" + str(Path(registry_dir).absolute()))
    mlflow.set_experiment(experiment_name=experiment_name)


def mlflow_dashboard(experiment_dir: PathLike):
    command = f"mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri file://{str(experiment_dir)} &"
    os.system(command)


def log_mlflow_params(params):
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)
