import logging
import socket
import subprocess
import tempfile
import time
import webbrowser
from pathlib import Path

import mlflow
import pandas as pd
import shap
import yaml

from autorad.config import config
from autorad.models.classifier import MLClassifier
from autorad.utils import io

log = logging.getLogger(__name__)


def get_model_by_name(name, models):
    for model in models:
        if model.name == name:
            return model
    raise ValueError(f"Model with name {name} not found")


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
    else:
        log.warning("Unable to start mlflow server: port 8000 is not open")
    time.sleep(2)
    webbrowser.open_new_tab("http://localhost:8000/")


def copy_artifacts_from(run_id: str):
    """
    Copy logged artfacts from a run to the current run
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=tmp_dir)
        for artifact in Path(tmp_dir).iterdir():
            mlflow.log_artifact(str(artifact), "feature_extraction")


def log_splits(splits: dict):
    with tempfile.TemporaryDirectory() as tmpdir:
        splits_path = Path(tmpdir) / "splits.json"
        io.save_json(splits, str(splits_path))
        mlflow.log_artifacts(str(splits_path))


def log_shap(model: MLClassifier, X_train: pd.DataFrame):
    explainer = shap.Explainer(model.predict_proba_binary, X_train)
    mlflow.shap.log_explainer(explainer, "shap-explainer")


def log_mlflow_params(params):
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)
