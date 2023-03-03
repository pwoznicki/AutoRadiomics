import logging
import socket
import subprocess
import tempfile
import time
import webbrowser
from pathlib import Path

import mlflow

from autorad.config import config
from autorad.utils import io

log = logging.getLogger(__name__)


def get_experiment_id_from_name(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"No experiment named {experiment_name} found. "
            "Please run the training first."
        )
    experiment_id = experiment.experiment_id
    return experiment_id


def get_best_run(experiment_id):
    all_runs = mlflow.search_runs(
        experiment_ids=experiment_id, order_by=["metrics.AUC_val"]
    )
    try:
        best_run = all_runs.iloc[-1]
    except IndexError:
        raise IndexError(
            "No trained models found. Please run the training first."
        )
    return best_run


def copy_artifacts_from(run_id: str):
    """
    Copy logged artfacts from a run to the current run
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=tmp_dir)
        for artifact in Path(tmp_dir).iterdir():
            mlflow.log_artifact(str(artifact), "feature_extraction")


def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def start_mlflow_server():
    mlflow_model_dir = "file://" + config.MODEL_REGISTRY

    port = 8000
    while not is_port_open(port):
        log.warning(
            f"Unable to start MLFlow server: port {port} is already in use"
        )
        port += 1
    subprocess.Popen(
        [
            "mlflow",
            "server",
            "-h",
            "0.0.0.0",
            "-p",
            str(port),
            "--backend-store-uri",
            mlflow_model_dir,
        ]
    )
    log.info(f"MLFlow server started successfully on port {port}")
    time.sleep(2)

    return port


def open_mlflow_dashboard(experiment_name="model_training", port=8000):
    experiment_id = get_experiment_id_from_name(experiment_name)
    url = f"http://localhost:{port}"
    if experiment_id is not None:
        url = f"{url}/#/experiments/{experiment_id}"
    webbrowser.open(url)


def log_dict_as_artifact(data: dict, artifact_name: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / f"{artifact_name}.yaml"
        io.save_yaml(data, tmp_path)
        mlflow.log_artifact(str(tmp_path))
