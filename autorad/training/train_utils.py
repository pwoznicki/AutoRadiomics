import logging
import socket
import subprocess
import time
import webbrowser

import mlflow
import shap

from autorad.config import config

log = logging.getLogger(__name__)


def log_shap(model, X_train):
    explainer = shap.Explainer(model, X_train)
    mlflow.shap.log_explainer(explainer, "shap-explainer")


def get_model_by_name(name, models):
    for model in models:
        if model.name == name:
            return model
    raise ValueError(f"Model with name {name} not found")


def log_mlflow_params(params):
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)


def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def start_mlflow_server():
    mlflow_model_dir = config.MODEL_REGISTRY
    if is_port_open(8000):
        subprocess.run(
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
    time.sleep(1)
    webbrowser.open_new_tab("http://localhost:8000/")
