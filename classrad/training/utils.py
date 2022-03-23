import os

import mlflow


def init_mlflow(experiment_name):
    # with mlflow.start_run(nested=True) as run:  # NOQA: F841
    #     run_id = mlflow.active_run().info.run_id
    #     print(f"MLflow run id: {run_id}")
    mlflow.set_experiment(experiment_name=experiment_name)


def mlflow_dashboard():
    os.system(
        "mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri $PWD/experiments/ &"
    )


def log_mlflow_params(params):
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)
