import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import shap

from autorad.data import FeatureDataset
from autorad.models import MLClassifier
from autorad.utils import io, mlflow_utils


def get_model_by_name(name, models):
    for model in models:
        if model.name == name:
            return model
    raise ValueError(f"Model with name {name} not found")


def log_splits(splits: dict):
    mlflow_utils.log_dict_as_artifact(splits, "splits")


def log_shap(model: MLClassifier, X_train: pd.DataFrame):
    explainer = shap.Explainer(model.predict_proba_binary, X_train)
    mlflow.shap.log_explainer(explainer, "shap-explainer")


def log_mlflow_params(params):
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)


def log_dataset(dataset: FeatureDataset):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        dataset.save(tmp_dir)
        mlflow.log_artifacts(tmp_dir, "dataset")
