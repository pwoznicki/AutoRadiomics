import logging
import socket
import subprocess
import tempfile
import time
import webbrowser
from pathlib import Path

import mlflow

from autorad.config import config

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
        experiment_ids=experiment_id, order_by=["metrics.AUC"]
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
    mlflow_model_dir = config.MODEL_REGISTRY
    if is_port_open(8000):
        subprocess.Popen(
            [
                "mlflow",
                "server",
                "-h",
                "0.0.0.0",
                "-p",
                "8000",
                "--backend-store-uri",
                mlflow_model_dir,
            ]
        )
        log.info("mlflow server started successfully")
        time.sleep(2)
    else:
        log.warning("Unable to start mlflow server: port 8000 is not open")


def open_mlflow_dashboard(experiment_name="model_training"):
    experiment_id = get_experiment_id_from_name(experiment_name)
    url = "http://localhost:8000"
    if experiment_id is not None:
        url = f"{url}/#/experiments/{experiment_id}"
    webbrowser.open_new_tab(url)
